"""Export the user-selected USS checkpoint without changing training sources.

Run in an environment with the original repo dependencies, onnx and onnxruntime.
Only CPU is used. The output contains the deployable graph, contract, and fixtures.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConfigData:
    pass

class SparseExportMoE(nn.Module):
    """Gather the selected expert's weights once per actual routing group."""
    def __init__(self, original):
        super().__init__()
        self.gate = original.gate.linear
        self.event_route = original.event_route
        self.register_buffer('w1', torch.stack([e[0].weight.T.detach() for e in original.experts]))
        self.register_buffer('b1', torch.stack([e[0].bias.detach() for e in original.experts]))
        self.register_buffer('w2', torch.stack([e[3].weight.T.detach() for e in original.experts]))
        self.register_buffer('b2', torch.stack([e[3].bias.detach() for e in original.experts]))
        assert original.n_shared_experts in (None, 0)
        self.has_shared_bias = original.n_shared_experts == 0
        if self.has_shared_bias:
            self.register_buffer('shared_bias', original.shared_experts[3].bias.detach().clone())

    def forward(self, x, event_labels=None):
        b, p, q, c = x.shape
        pool = x.mean(dim=(1, 2)) if self.event_route else x.mean(dim=2).reshape(b*p, c)
        self.probs = self.gate(pool).softmax(dim=-1)
        selected_weight, selected = torch.topk(self.probs, 1, dim=-1, sorted=False)
        self.selected = selected.squeeze(-1)
        self.selected_weight = selected_weight.squeeze(-1)
        groups = b if self.event_route else b*p
        tokens = x.reshape(groups, -1, c)
        z = torch.bmm(tokens, self.w1[self.selected]) + self.b1[self.selected].unsqueeze(1)
        z = z * torch.sigmoid(z)
        z = torch.bmm(z, self.w2[self.selected]) + self.b2[self.selected].unsqueeze(1)
        z = z * selected_weight.unsqueeze(-1)
        z = z.reshape(b, p, q, c)
        return (z + self.shared_bias) * .5 if self.has_shared_bias else z

class PackedDecoder(nn.Module):
    """Merge overlapping bands using fixed gather indices instead of slice updates."""
    def __init__(self, original):
        super().__init__()
        self.mlp_mask = original.mlp_mask
        self.mlp_residual = original.mlp_residual
        self.bands = [(int(lo), int(hi)) for lo, hi in zip(original.subbands_low, original.subbands_up)]
        self.num_spk = original.num_spk
        self.freq_dim = int(original.freq_dim)
        covers = [[] for _ in range(self.freq_dim)]
        offset = 0
        for lo, hi in self.bands:
            for f in range(lo, hi):
                covers[f].append(offset + f - lo)
            offset += hi-lo
        assert all(covers)
        self.max_overlap = max(map(len, covers))
        indices = [[covers[f][o] if o < len(covers[f]) else offset for f in range(self.freq_dim)] for o in range(self.max_overlap)]
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long).flatten())
        self.register_buffer('counts', torch.tensor([len(c) for c in covers], dtype=torch.float32).reshape(1,1,1,-1,1))

    def merge(self, bands):
        joined = torch.cat(bands, dim=3)
        joined = torch.cat([joined, torch.zeros_like(joined[:, :, :, :1])], dim=3)
        b,t,s,_,ri = joined.shape
        merged = joined.index_select(3, self.indices).reshape(b,t,s,self.max_overlap,self.freq_dim,ri).sum(3) / self.counts
        return merged.transpose(1, 2)

    def forward(self, x):
        b,c,t,k = x.shape
        masks, residuals = [], []
        for i,(lo,hi) in enumerate(self.bands):
            band=x[:,:,:,i]
            masks.append(self.mlp_mask[i](band).transpose(1,2).reshape(b,t,self.num_spk,hi-lo,2))
            residuals.append(self.mlp_residual[i](band).transpose(1,2).reshape(b,t,self.num_spk,hi-lo,2))
        return self.merge(masks), self.merge(residuals)

class ExportCore(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, spectrum):
        x = self.core.melband_split(spectrum)
        for block in self.core.blocks:
            x = block(x)
        mask, residual = self.core.melmask_decoder(x)
        real = mask[...,0] * spectrum[:,None,...,0] - mask[...,1] * spectrum[:,None,...,1] + residual[...,0]
        imag = mask[...,0] * spectrum[:,None,...,1] + mask[...,1] * spectrum[:,None,...,0] + residual[...,1]
        separated = torch.stack([real,imag], dim=-1)
        event, event_ids, event_weight, frame_ids, band_ids, frame_weight, band_weight = [],[],[],[],[],[],[]
        for block in self.core.blocks:
            for module in (block.f_module, block.t_module):
                router = module.conformer.ff1.fn.fn
                event.append(router.probs[0]); event_ids.append(router.selected[0]); event_weight.append(router.selected_weight[0])
            fg=block.f_module.conformer.ff2.fn.fn
            tg=block.t_module.conformer.ff2.fn.fn
            frame_ids.append(fg.selected); band_ids.append(tg.selected)
            frame_weight.append(fg.selected_weight); band_weight.append(tg.selected_weight)
        return (separated, torch.stack(event), torch.stack(event_ids), torch.stack(event_weight), torch.stack(frame_ids), torch.stack(band_ids), torch.stack(frame_weight), torch.stack(band_weight))

OUTPUT_NAMES=['separated_spectrum','event_probs','event_selected','event_weight','frame_selected','band_selected','frame_weight','band_weight']

def error(actual, expected):
    delta=np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64)
    mse=np.mean(delta**2); energy=np.mean(np.asarray(expected,dtype=np.float64)**2)
    return {'max_abs':float(np.max(np.abs(delta))), 'mean_abs':float(np.mean(np.abs(delta))), 'snr_db':float(10*np.log10((energy+1e-30)/(mse+1e-30)))}
