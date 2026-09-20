"""Export the same EMA model with a dynamic frame axis (3-30 s)."""
import argparse,copy,hashlib,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from export_helpers import ConfigData,SparseExportMoE,PackedDecoder,ExportCore,OUTPUT_NAMES,error

def tiled_attention(self, query, key, value, attn_bias=None, p=0.0):
    assert attn_bias is None and p == 0
    q=(query*self.scale).transpose(1,2)
    k=key.transpose(1,2).transpose(-2,-1)
    v=value.transpose(1,2)
    n=q.shape[2]
    # Bound the largest attention matrix for a 30-second browser run.
    parts=[]
    for i in range(8):
        scores=q[:,:,n*i//8:n*(i+1)//8] @ k
        parts.append(scores.softmax(-1) @ v)
    return torch.cat(parts,dim=2).transpose(1,2).contiguous()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--audio',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    import onnx,onnxruntime as ort,soundfile as sf
    sys.path.insert(0,args.repo);torch.set_num_threads(2)
    import baseline_code.models.conformer as conformer
    conformer.xops=None
    from baseline_code.models.TF_MOE import TF_MOE_SE
    with torch.serialization.safe_globals([(ConfigData,'baseline_code.config.Config')]): ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=vars(ckpt['hyper_parameters']['cfg']);mc=cfg['model_configs'];print('CONFIG',json.dumps(mc), 'EPOCH',ckpt.get('epoch'),flush=True);original=TF_MOE_SE(**mc)
    assert mc['n_expert']==mc['event_num_experts']==12 and mc['num_spk']==3
    assert mc['n_fft']==960 and mc['hop_length']==480 and mc['target_fs']==16000
    original.load_state_dict({k.removeprefix('se_model.'):v for k,v in ckpt['state_dict'].items()},strict=True)
    named=list(original.named_parameters());shadows=ckpt['ema']['shadow_params'];assert len(named)==len(shadows)
    for (name,param),shadow in zip(named,shadows):
        assert param.shape==shadow.shape,name
        param.data.copy_(shadow)
    original.eval();adapted=copy.deepcopy(original)
    adapted.se.melmask_decoder=PackedDecoder(adapted.se.melmask_decoder)
    for block in adapted.se.blocks:
        for module in (block.f_module,block.t_module):
            for ff in (module.conformer.ff1,module.conformer.ff2):ff.fn.fn=SparseExportMoE(ff.fn.fn)
        # Only temporal attention has a duration-dependent, quadratic matrix.
        for module in block.t_module.modules():
            if hasattr(module,'native_attention'):module.native_attention=types.MethodType(tiled_attention,module)
    for module in adapted.modules():
        if hasattr(module,'cache_if_possible'):module.cache_if_possible=False
    core=ExportCore(adapted.se).eval();samples,sr=sf.read(args.audio,dtype='float32');assert sr==16000 and samples.ndim==1
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def spec_of(wave):return torch.view_as_real(torch.stft(torch.from_numpy(wave)[None],n_fft=960,hop_length=480,window=torch.hann_window(960),return_complex=True).transpose(1,2)).contiguous()
    with torch.inference_mode():
        spec=spec_of(samples[:96000])
        torch.onnx.export(core,(spec,),str(out/'uss.onnx'),input_names=['spectrum'],output_names=OUTPUT_NAMES,opset_version=17,dynamo=False,do_constant_folding=True,dynamic_axes={'spectrum':{1:'frames'},'separated_spectrum':{2:'frames'},'frame_selected':{1:'frames'},'frame_weight':{1:'frames'}})
    onnx.checker.check_model(onnx.load(str(out/'uss.onnx')))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(out/'uss.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    report={'checkpointSha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'cases':[],'dynamicAxes':True,'attentionQueryTiles':8,'tolerances':{'spectralSnrDbMin':75,'eventProbabilityMaxAbs':1e-4,'tfReferenceProbabilityGapMax':1e-7},'modelConfig':mc,'checkpointEpoch':ckpt.get('epoch'),'checkpointStep':ckpt.get('global_step')}
    for duration,start in [(3,0),(6,0),(10,0),(7.13,1.17),(30,0)]:
        length=round(duration*sr);signal=np.tile(samples,4)[round(start*sr):round(start*sr)+length].copy();spec=spec_of(signal);events=[];tf_events=[]
        hooks=[]
        for block in original.se.blocks:
            for module in (block.f_module,block.t_module):
                gate=module.conformer.ff1.fn.fn.gate
                hooks.append(gate.register_forward_hook(lambda m,ins,outs:events.append(m.linear(ins[0].mean((1,2))).softmax(-1).detach().clone())))
        for block in original.se.blocks:
            for module in (block.f_module,block.t_module):
                hooks.append(module.conformer.ff2.fn.fn.gate.register_forward_hook(lambda m,ins,outs:tf_events.append(outs[0].detach().clone())))
        with torch.inference_mode():
            reference=torch.view_as_real(original.se(spec)).numpy()
            expected=[x.numpy() for x in core(spec)]
        for h in hooks:h.remove()
        started=time.perf_counter();actual=session.run(None,{'spectrum':spec.numpy()});elapsed=time.perf_counter()-started
        metric=error(actual[0],reference);prob=error(actual[1],torch.cat(events).numpy())
        assert metric['snr_db']>75,metric
        print('NUMERICS',duration,json.dumps({'spectrum':metric,'eventProbs':prob,'adaptedSpectrum':error(expected[0],reference),'adaptedEventProbs':error(expected[1],torch.cat(events).numpy())}),flush=True)
        assert prob['max_abs']<1e-4,prob
        assert np.array_equal(actual[2],np.argmax(torch.cat(events).numpy(),-1))
        tf_checks={}
        for axis,idx,offset in [('frame',4,0),('band',5,1)]:
            original_ids=np.stack([x.numpy().reshape(-1) for x in tf_events[offset::2]])
            assert np.array_equal(expected[idx],original_ids), 'Adapter changed original TF routes'
            mismatch=np.argwhere(actual[idx]!=original_ids)
            gaps=[]
            for layer,token in mismatch:
                module=getattr(adapted.se.blocks[layer], 'f_module' if axis=='frame' else 't_module')
                probs=module.conformer.ff2.fn.fn.probs.detach().numpy()[token]
                gaps.append(float(probs[original_ids[layer,token]]-probs[actual[idx][layer,token]]))
            tf_checks[axis]={'total':int(original_ids.size),'mismatches':len(mismatch),'maxReferenceProbabilityGap':max(gaps,default=0),'examples':mismatch[:10].tolist()}
        print('TF CHECKS',duration,json.dumps(tf_checks),flush=True)
        assert all(v['maxReferenceProbabilityGap']<=1e-7 for v in tf_checks.values()),tf_checks
        n=spec.shape[1]
        wav=torch.istft(torch.view_as_complex(torch.from_numpy(actual[0]).contiguous()).reshape(3,n,481).transpose(1,2),n_fft=960,hop_length=480,window=torch.hann_window(960),length=length).numpy()
        assert np.isfinite(wav).all()
        case={'seconds':duration,'start':start,'frames':n,'spectralError':metric,'eventProbabilityError':prob,'adaptedSpectralError':error(expected[0],reference),'adaptedEventProbabilityError':error(expected[1],torch.cat(events).numpy()),'eventIdsMatch':True,'tfIdsMatch':all(v['mismatches']==0 for v in tf_checks.values()),'tfRouteComparison':tf_checks,'runtimeSeconds':elapsed};report['cases'].append(case);print(json.dumps(case),flush=True)
        slug=str(duration).replace('.','p');signal.astype('<f4').tofile(out/f'input-{slug}.f32');wav.astype('<f4').tofile(out/f'output-{slug}.f32');(out/f'routes-{slug}.json').write_text(json.dumps({k:v.tolist() for k,v in zip(OUTPUT_NAMES[1:],actual[1:])}))
    model=(out/'uss.onnx').read_bytes()
    manifest={'schemaVersion':2,'modelId':'uss-moe-large-event-last-12experts-ema-fp32-dynamic','model':{'url':'uss.onnx','bytes':len(model),'sha256':hashlib.sha256(model).hexdigest()},'sampleRate':16000,'minSamples':48000,'maxSamples':480000,'fftSize':960,'hopLength':480,'window':'hann-periodic','center':True,'padMode':'reflect','input':{'name':'spectrum','shape':[1,'frames',481,2],'dtype':'float32'},'outputs':OUTPUT_NAMES,'layers':mc['num_layer'],'experts':mc['n_expert'],'eventExperts':mc['event_num_experts'],'tfExperts':mc['n_expert'],'sources':mc['num_spk'],'bands':mc['num_bands'],'channels':mc['num_channel'],'weightVariant':'EMA','sharedExperts':mc.get('n_shared_experts'),'useGroupNorm':mc.get('use_group_norm',False),'checkpointEpoch':ckpt.get('epoch'),'checkpointStep':ckpt.get('global_step'),'expertLabels':['English','Mandarin · AISHELL-1','Mandarin · AISHELL-3','Other speech','Singing / vocals','Other human sounds','Birds','Other animals','String instruments','Other music','Transport','Nature / other'],'modules':[f'L{i+1} {axis}' for i in range(mc['num_layer']) for axis in ('F','T')],'bandLowHz':original.se.melband_split.subband_freqs_low.tolist(),'bandHighHz':original.se.melband_split.subband_freqs_up.tolist(),'paddingPolicy':'No fixed-length zero padding; centered STFT reflect padding only','checkpointSha256':report['checkpointSha256'],'attentionQueryTiles':8}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'validation.json').write_text(json.dumps(report,indent=2));print('EXPORT COMPLETE',len(model),flush=True)
if __name__=='__main__':main()
