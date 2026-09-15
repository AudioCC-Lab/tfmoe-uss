// Plot the exact input spectrum and the selected layer's actual FFN2 routes.
export const TIME_COLORS = ['#4e9aca','#ff963d','#72ba79','#e56669','#a28aca','#b9947c'];
export const BAND_COLORS = ['#80cdb1','#f49772','#d797bd','#b6d65f','#e4c58e','#b5bbc2'];
const stops = [[0,[0,0,4]],[.13,[31,12,72]],[.25,[87,16,110]],[.4,[148,38,103]],[.55,[203,65,73]],[.7,[240,112,31]],[.85,[250,176,15]],[1,[252,255,164]]];
const palette = Array.from({length:256},(_,i)=>{
  const v=i/255,j=stops.findIndex(s=>s[0]>=v),hi=stops[Math.max(1,j)],lo=stops[Math.max(0,j-1)],a=(v-lo[0])/(hi[0]-lo[0]);
  return lo[1].map((x,k)=>Math.round(x+(hi[1][k]-x)*a));
});
const rgb = hex => [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
const paint = (image,x,y,color) => {const i=(y*image.width+x)*4;image.data.set([...color,255],i);};

export function createTFPlot({spectrum,manifest:m,routes,duration,clipStart}) {
  const frames=m.input.shape[1],bins=m.input.shape[2],hopSeconds=m.hopLength/m.sampleRate,nyquist=m.sampleRate/2;
  const width=Math.max(1,Math.min(frames,Math.floor(duration/hopSeconds+1e-7)+1));
  const frameAt=x=>Math.min(width-1,Math.round(x));
  const centers=m.bandLowHz.map((lo,i)=>(lo+m.bandHighHz[i])/2);
  const bandAtHz=hz=>{let best=0;for(let i=1;i<centers.length;i++)if(Math.abs(centers[i]-hz)<Math.abs(centers[best]-hz))best=i;return best;};
  const el=document.createElement('div');el.className='spectrogram-view';
  el.innerHTML=`<div class="spec-heading"><h3>Mixture spectrogram</h3><span>${clipStart.toFixed(2)}–${(clipStart+duration).toFixed(2)} s in source</span></div><div class="spec-legend"><span class="spec-legend-name">Time · F module</span>${TIME_COLORS.map((c,i)=>`<span><i style="background:${c}"></i>E${i}</span>`).join('')}</div><div class="spec-legend"><span class="spec-legend-name">Bands · T module</span>${BAND_COLORS.map((c,i)=>`<span><i style="background:${c}"></i>E${i}</span>`).join('')}</div><div class="spec-grid"><span class="spec-hz">Hz</span><div class="spec-frame-axis"><span>Frame 0</span><span>Frame ${Math.floor((width-1)/2)}</span><span>Frame ${width-1}</span></div><div class="spec-frequency-axis">${[1,.75,.5,.25,0].map(f=>`<span>${f*nyquist/1000}${f?'k':''}</span>`).join('')}</div><canvas id="mixture-spectrogram" class="spec-main" role="img" aria-label="Actual input mixture spectrogram, ${duration.toFixed(2)} seconds, 0 to ${nyquist} Hz"></canvas><div class="spec-band-wrap"><canvas id="band-route-strip" tabindex="0" role="slider" aria-label="Frequency-band routing: use up and down arrow keys" aria-valuemin="0" aria-valuemax="${centers.length-1}" aria-valuenow="0"></canvas></div><span class="spec-time-label">Time</span><canvas id="time-route-strip" tabindex="0" role="slider" aria-label="Time-frame routing: use left and right arrow keys" aria-valuemin="0" aria-valuemax="${width-1}" aria-valuenow="0"></canvas><span class="spec-band-label">Bands</span><div class="spec-time-axis"><span>0 s</span><span>${(duration/2).toFixed(2)} s</span><span>${duration.toFixed(2)} s</span></div></div><div class="spec-scale"><span>−80 dB</span><i></i><span>0 dB</span><span>Relative magnitude</span></div><p class="spec-inspection" id="tf-inspection" aria-live="polite"></p><p class="hint spec-note">Bottom: time-frame routing from F modules. Right: band routing from T modules, mapped by nearest Mel-band center. Only the selected audio duration is shown; the model still uses a 6-second input with zero padding when needed.</p>`;
  const main=el.querySelector('#mixture-spectrogram'),time=el.querySelector('#time-route-strip'),band=el.querySelector('#band-route-strip');
  main.width=width;main.height=bins;time.width=width;time.height=22;band.width=24;band.height=bins;
  main.dataset.duration=String(duration);main.dataset.frames=String(width);main.dataset.bins=String(bins);
  const g=main.getContext('2d'),pixels=g.createImageData(width,bins);
  let peak=0;
  for(let x=0;x<width;x++)for(let f=0;f<bins;f++){const i=(frameAt(x)*bins+f)*2;peak=Math.max(peak,Math.hypot(spectrum[i],spectrum[i+1]));}
  for(let x=0;x<width;x++)for(let y=0;y<bins;y++){
    const i=(frameAt(x)*bins+bins-1-y)*2;
    const db=20*Math.log10(Math.max(1e-12,Math.hypot(spectrum[i],spectrum[i+1])/Math.max(1e-12,peak)));
    paint(pixels,x,y,palette[Math.max(0,Math.min(255,Math.round((db+80)/80*255)))]);
  }
  g.putImageData(pixels,0,0);
  let layer=0,frameCursor=0,bandCursor=0;
  function describe(announce=true) {
    const timeIndex=layer*frames+frameCursor,bandIndex=layer*centers.length+bandCursor;
    const t=`Time frame ${frameCursor} (${(frameCursor*hopSeconds).toFixed(2)} s): E${routes.frame_selected.data[timeIndex]}, ${(routes.frame_weight.data[timeIndex]*100).toFixed(1)}%`;
    const b=`Band ${bandCursor+1} (${Math.round(m.bandLowHz[bandCursor])}–${Math.round(m.bandHighHz[bandCursor])} Hz): E${routes.band_selected.data[bandIndex]}, ${(routes.band_weight.data[bandIndex]*100).toFixed(1)}%`;
    time.setAttribute('aria-valuenow',String(frameCursor));time.setAttribute('aria-valuetext',`L${layer+1} · ${t}`);
    band.setAttribute('aria-valuenow',String(bandCursor));band.setAttribute('aria-valuetext',`L${layer+1} · ${b}`);
    time.title=t;band.title=b;
    const detail=el.querySelector('#tf-inspection');detail.setAttribute('aria-live',announce?'polite':'off');
    detail.textContent=`L${layer+1} · ${t} · ${b}`;
  }
  function setLayer(index) {
    layer=Math.max(0,Math.min(m.layers-1,index));
    const tg=time.getContext('2d'),bg=band.getContext('2d'),bp=bg.createImageData(24,bins);
    for(let x=0;x<width;x++){tg.fillStyle=TIME_COLORS[routes.frame_selected.data[layer*frames+frameAt(x)]];tg.fillRect(x,0,1,22);}
    for(let y=0;y<bins;y++){
      const b=bandAtHz((bins-1-y)/(bins-1)*nyquist),color=rgb(BAND_COLORS[routes.band_selected.data[layer*centers.length+b]]);
      for(let x=0;x<24;x++)paint(bp,x,y,color);
    }
    bg.putImageData(bp,0,0);el.dataset.layer=String(layer);describe();
  }
  function locate(e,axis) {
    const rect=e.currentTarget.getBoundingClientRect();
    if(axis!=='band')frameCursor=frameAt(Math.max(0,Math.min(width-1,Math.round((e.clientX-rect.left)/rect.width*(width-1)))));
    if(axis!=='time')bandCursor=bandAtHz(Math.max(0,Math.min(1,1-(e.clientY-rect.top)/rect.height))*nyquist);
    describe(e.type!=='pointermove');
  }
  for(const [canvas,axis]of [[main,'both'],[time,'time'],[band,'band']]){canvas.onpointermove=e=>locate(e,axis);canvas.onclick=e=>locate(e,axis);}
  time.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();frameCursor=e.key==='Home'?0:e.key==='End'?width-1:Math.max(0,Math.min(width-1,frameCursor+(e.key==='ArrowRight'?1:-1)));describe();}};
  band.onkeydown=e=>{if(['ArrowUp','ArrowDown','Home','End'].includes(e.key)){e.preventDefault();bandCursor=e.key==='Home'?0:e.key==='End'?centers.length-1:Math.max(0,Math.min(centers.length-1,bandCursor+(e.key==='ArrowUp'?1:-1)));describe();}};
  setLayer(0);
  return {element:el,setLayer};
}
