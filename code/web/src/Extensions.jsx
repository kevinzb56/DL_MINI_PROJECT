import { useEffect, useState } from 'react';
import { ArrowUpRight, Download, Focus, Layers, ListFilter, PackageCheck, RefreshCw } from 'lucide-react';
import { api, post } from './api';

export function usePhase7(ctx) {
  const [info,setInfo]=useState(null);
  const refresh=async()=>{if(ctx.project)setInfo(await api(ctx.endpoint('/phase7')));};
  useEffect(()=>{let active=true;if(ctx.project)api(ctx.endpoint('/phase7')).then(value=>{if(active)setInfo(value);}).catch(()=>{if(active)setInfo(null);});return()=>{active=false;};},[ctx.project?.id]);
  return {info,refresh};
}

export function InferenceOptions({ctx,version,mode,runtime,onMode,onRuntime}) {
  const {info,refresh}=usePhase7(ctx);
  const ready=info?.onnx_exports?.filter(row=>row.version_id===version&&row.state==='verified').at(-1);
  useEffect(()=>{if(info&&!ready&&runtime==='onnx')onRuntime('pytorch');},[info,ready,version,runtime]);
  const exportModel=()=>ctx.run('Exporting ONNX and checking native output parity…',async()=>{
    const report=await api(ctx.endpoint(`/versions/${version}/onnx`),{method:'POST'});
    await refresh();
    ctx.notify(`ONNX probability parity passed (maximum difference ${report.max_probability_difference.toExponential(2)}). ${report.audited_source_validation?'Audited source-validation metric checks also passed.':'Dataset metric checks were not assessed: prepare source replay data first.'}`);
  });
  return <section className="phase7-options panel"><div className="extension-label"><Layers size={16}/><span>PHASE 07 <strong>Inference & portability</strong></span></div><label>IMAGE SAMPLING<select id="inferenceMode" value={mode} disabled={!!ctx.busy} onChange={e=>onMode(e.target.value)}><option value="resize">Whole-image resize</option><option value="tiled">Overlapping tiles · 256 px</option></select></label><label>RUNTIME<select id="runtimeSelect" value={runtime} disabled={!!ctx.busy} onChange={e=>onRuntime(e.target.value)}><option value="pytorch">PyTorch CPU</option><option value="onnx" disabled={!ready}>ONNX Runtime{ready?'':' · export first'}</option></select></label><button id="exportOnnx" className="btn secondary" disabled={!!ctx.busy||!version} onClick={exportModel}><PackageCheck size={14}/> Export & verify ONNX</button>{ready&&<a id="downloadOnnx" className="text-button" href={ctx.endpoint(`/onnx/${ready.id}/download`)}><Download size={13}/> Download ONNX ZIP</a>}<p>{mode==='tiled'?'Tiles preserve spatial scale, overlap by 64 pixels, and blend probabilities before thresholding. This is a separate inference mode, not an automatic accuracy improvement.':'Whole-image mode preserves the original experiment preprocessing. Export packages include an input contract, graph checksum, and parity report.'}</p></section>;
}

export function ReplayOption({ctx,value,onChange}) {
  const {info}=usePhase7(ctx);
  const ready=!!info?.source_replay?.ready;
  return <div className="replay-option"><label><input id="sourceReplay" type="checkbox" checked={value} disabled={!ready||!!ctx.busy} onChange={e=>onChange(e.target.checked)}/><span><strong>Mix in source replay</strong><small>{ready?`${info.source_replay.source_train_images} verified source-training pairs · one source image per support image`:'Prepare verified source data with scripts/prepare_source_replay.py'}</small></span></label><p>{value?'Adds training examples and compute without changing the support-pass budget. A separate 8-image source-validation subset also gates activation; this is not the full source-holdout score.':'Optional replay uses source training only. Source-validation images are reserved for evaluation and never enter optimizer updates.'}</p></div>;
}

export function Acquisition({ctx,version}) {
  const [policy,setPolicy]=useState('entropy'),[ranking,setRanking]=useState(null);
  useEffect(()=>setRanking(null),[ctx.project?.id,version]);
  const rank=()=>ctx.run('Ranking eligible correction images…',async()=>setRanking(await api(ctx.endpoint('/rankings'),post({version_id:version,policy,seed:7301}))));
  return <section className="panel acquisition-panel"><div className="section-heading"><div><div className="eyebrow">SUGGESTIONS, WITH A HUMAN CHOICE</div><h2>Where could a correction help?</h2></div><ListFilter size={21}/></div><p>Rank only eligible incomplete or unlabeled images. Frozen evaluation images, complete corrections, and job validation images stay outside the pool.</p><div className="acquisition-actions"><label>SELECTION POLICY<select id="rankingPolicy" value={policy} disabled={!!ctx.busy} onChange={e=>setPolicy(e.target.value)}><option value="entropy">Predictive entropy</option><option value="random">Seeded random · comparison</option></select></label><button id="rankPool" className="btn secondary" disabled={!!ctx.busy||!version} onClick={rank}><RefreshCw size={14}/> Rank correction pool</button></div>{ranking&&<div id="rankingResults"><div className="ranking-caption">{ranking.pool_size} eligible images · {ranking.policy} · seed {ranking.seed}</div>{ranking.ranked.length===0?<p>No eligible unlabeled images remain.</p>:ranking.ranked.map((row,index)=><div className="ranking-row" key={row.image_id}><span>{String(index+1).padStart(2,'0')}</span><strong>{row.name}</strong><small>{row.score===null?'Seeded order':`${row.score.toFixed(4)} bits`}</small></div>)}</div>}<div className="small-note"><Focus size={15}/><span>Entropy is a diagnostic uncertainty score, not calibrated confidence. Confident predictions can be wrong. Review suggestions manually; no support images are chosen automatically.</span></div></section>;
}
