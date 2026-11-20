import { useEffect, useRef, useState } from 'react';
import { ArrowRight, ArrowUpRight, Brush, Check, Crosshair, Download, Eraser, Expand, ImagePlus, Layers, Play, Plus, Redo2, Save, ScanLine, SlidersHorizontal, Undo2, UploadCloud, X, ZoomIn, ZoomOut } from 'lucide-react';
import { api, loadImage, post } from './api';
import { InferenceOptions } from './Extensions';

function Viewer({ title, tag, children, loaded, id, editor=false, zoom, image, sync, stageRef }) {
  const ref=useRef(null);
  useEffect(()=>{
    const fit=()=>{
      const node=ref.current;if(!node||!image)return;
      const media=node.querySelector('canvas,img');if(!media)return;
      const scale=Math.min(node.clientWidth/image.width,node.clientHeight/image.height)*zoom;
      media.style.width=`${Math.max(1,Math.round(image.width*scale))}px`;
      media.style.height=`${Math.max(1,Math.round(image.height*scale))}px`;
    };
    fit();const observer=new ResizeObserver(fit);if(ref.current)observer.observe(ref.current);
    return()=>observer.disconnect();
  },[image,zoom,loaded]);
  return <article className={`viewer-card ${editor?'correction-card':''}`}><header><span><i className={`model-dot ${tag==='ORIGINAL'?'gray':''}`}/>{title}</span><small>{tag}</small></header><div ref={node=>{ref.current=node;if(stageRef)stageRef.current=node;}} id={id} className={`viewport ${loaded?'loaded':''}`} onScroll={sync}>{children}{!loaded&&<div className="viewer-empty"><ScanLine size={27}/><strong>{editor?'Your expertise, pixel by pixel.':'A closer look starts here.'}</strong><span>{editor?'Paint cracks. Refine the boundaries.':'Add an image, then run a prediction.'}</span></div>}</div></article>;
}

export default function Studio({ ctx, mode }) {
  const { project, images, versions, busy, endpoint, run, notify, refresh, setDirty, navigate, health }=ctx;
  const [selected,setSelected]=useState(null),[original,setOriginal]=useState(null),[prediction,setPrediction]=useState(null),[comparison,setComparison]=useState(null);
  const [version,setVersion]=useState(''),[compareVersion,setCompareVersion]=useState('');
  const [inferenceMode,setInferenceMode]=useState('resize'),[runtime,setRuntime]=useState('pytorch');
  const [threshold,setThreshold]=useState(.5),[opacity,setOpacity]=useState(.5),[display,setDisplay]=useState('overlay'),[zoom,setZoom]=useState(1);
  const [tool,setTool]=useState('brush'),[brushSize,setBrushSize]=useState(8),[complete,setComplete]=useState(false),[saved,setSaved]=useState(false),[isDirty,setIsDirty]=useState(false),[history,setHistory]=useState({undo:0,redo:0});
  const editor=useRef(null),mask=useRef(null),undo=useRef([]),redo=useRef([]),drawing=useRef(null),scrolling=useRef(false),upload=useRef(null);
  const originalStage=useRef(null),predictionStage=useRef(null),editorStage=useRef(null);
  useEffect(()=>{if(!version&&versions.length)setVersion(project?.active_version||versions[0].id);if(!compareVersion&&versions.length)setCompareVersion(versions.find(v=>v.phase===3)?.id||versions[0].id);},[versions,project,version,compareVersion]);
  useEffect(()=>{
    if(project?.active_version){
      setVersion(project.active_version);
      setPrediction(null);
      setComparison(null);
    }
  },[project?.active_version]);
  const sync=event=>{if(scrolling.current)return;scrolling.current=true;for(const ref of [originalStage,predictionStage,editorStage])if(ref.current&&ref.current!==event.currentTarget){ref.current.scrollLeft=event.currentTarget.scrollLeft;ref.current.scrollTop=event.currentTarget.scrollTop;}requestAnimationFrame(()=>scrolling.current=false);};
  const markDirty=()=>{setIsDirty(true);setDirty(true);setSaved(false);};
  const updateHistory=()=>setHistory({undo:undo.current.length,redo:redo.current.length});
  const remember=()=>{const canvas=mask.current;undo.current.push(canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height));const limit=Math.max(1,Math.min(12,Math.floor(64*1024*1024/(canvas.width*canvas.height*4))));while(undo.current.length>limit)undo.current.shift();redo.current=[];updateHistory();};
  const binarize=()=>{const canvas=mask.current,context=canvas.getContext('2d');const data=context.getImageData(0,0,canvas.width,canvas.height);for(let i=0;i<data.data.length;i+=4){const value=data.data[i]>=128?255:0;data.data[i]=data.data[i+1]=data.data[i+2]=value;data.data[i+3]=255;}context.putImageData(data,0,0);};
  function drawEditor(){
    if(!original||!editor.current||!mask.current||mode==='compare')return;
    const canvas=editor.current;canvas.width=original.width;canvas.height=original.height;
    const context=canvas.getContext('2d');context.drawImage(original,0,0);
    const pixels=mask.current.getContext('2d').getImageData(0,0,canvas.width,canvas.height);
    for(let i=0;i<pixels.data.length;i+=4){const active=pixels.data[i]>127;pixels.data[i]=247;pixels.data[i+1]=144;pixels.data[i+2]=72;pixels.data[i+3]=active?Math.round(opacity*255):0;}
    const tint=document.createElement('canvas');tint.width=canvas.width;tint.height=canvas.height;tint.getContext('2d').putImageData(pixels,0,0);context.drawImage(tint,0,0);
  }
  useEffect(()=>{drawEditor();},[original,opacity,mode]);
  async function openImage(row, discardConfirmed=false){
    if(!discardConfirmed&&isDirty&&!window.confirm('Discard unsaved correction changes?'))return;
    const image=await loadImage(endpoint(`/images/${row.id}/original`));
    const canvas=document.createElement('canvas');canvas.width=row.width;canvas.height=row.height;const context=canvas.getContext('2d',{willReadFrequently:true});context.fillStyle='black';context.fillRect(0,0,row.width,row.height);
    if(row.correction)context.drawImage(await loadImage(endpoint(`/corrections/${row.correction.id}/mask`)),0,0);
    mask.current=canvas;undo.current=[];redo.current=[];setHistory({undo:0,redo:0});setSelected(row);setOriginal(image);setPrediction(null);setComparison(null);setComplete(!!row.correction?.complete);setSaved(!!row.correction);setIsDirty(false);setDirty(false);setZoom(1);
    if(!row.training_eligible)notify('This image matches frozen evaluation data. It can be inspected but cannot enter local adaptation or validation.',true);
  }
  async function addFiles(files){
    if(!project||!files.length)return;
    if(isDirty&&!window.confirm('Discard unsaved correction changes?'))return;
    await run('Adding road images…',async()=>{let last;const errors=[];for(const file of files){const body=new FormData();body.append('file',file);try{last=await api(endpoint('/images'),{method:'POST',body});}catch(error){errors.push(`${file.name}: ${error.message}`);}}await refresh();if(last){setIsDirty(false);await openImage(last,true);}if(errors.length)notify(errors.join(' · '),true);});
  }
  const invalidate=()=>{setPrediction(null);setComparison(null);};
  const requestPrediction=id=>api(endpoint('/predict'),post({image_id:selected.id,version_id:id,threshold,opacity,inference_mode:inferenceMode,runtime}));
  const predict=()=>run(mode==='compare'?'Comparing model versions…':'Running local inference…',async()=>{if(!selected)return;setPrediction(await requestPrediction(version));if(mode==='compare')setComparison(await requestPrediction(compareVersion));});
  const initializeFromPrediction=()=>run('Loading predicted mask…',async()=>{remember();const image=await loadImage(endpoint(`/predictions/${prediction.id}/mask`));mask.current.getContext('2d').drawImage(image,0,0);binarize();setComplete(false);markDirty();drawEditor();});
  const suggestMissedCracks=()=>run('Asking vision agent to inspect missed cracks…',async()=>{
    if(!health?.ai_ready){notify(health?.ai_message||'AI suggestions are not configured. Restart CrackLab with a Groq API key.',true);return;}
    const result=await api(endpoint(prediction?`/predictions/${prediction.id}/ai-suggestions`:`/images/${selected.id}/ai-suggestions`),{method:'POST'});
    remember();
    if(prediction){const modelMask=await loadImage(endpoint(`/predictions/${prediction.id}/mask`));const context=mask.current.getContext('2d');context.clearRect(0,0,selected.width,selected.height);context.drawImage(modelMask,0,0);}
    const context=mask.current.getContext('2d');
    if(!result.paths.length){binarize();setComplete(false);markDirty();drawEditor();notify(`Model mask loaded. Groq found no additional paths to suggest.`);return;}
    context.strokeStyle='white';context.lineCap='round';context.lineJoin='round';
    for(const path of result.paths){
      context.lineWidth=Math.max(1,path.width*Math.min(selected.width,selected.height)/1000);
      context.beginPath();path.points.forEach(([x,y],index)=>{const px=x*selected.width/1000,py=y*selected.height/1000;if(index)context.lineTo(px,py);else context.moveTo(px,py);});context.stroke();
    }
    binarize();setComplete(false);markDirty();drawEditor();
    notify(`${result.paths.length} AI-suggested crack paths added as an unsaved draft. Review and erase errors before saving.`);
  });
  const save=()=>run('Saving your correction…',async()=>{
    binarize();const blob=await new Promise(resolve=>mask.current.toBlob(resolve,'image/png'));const body=new FormData();body.append('file',blob,'mask.png');
    const correction=await api(endpoint(`/images/${selected.id}/corrections?complete=${complete}`),{method:'POST',body});
    setSelected({...selected,correction});setIsDirty(false);setDirty(false);setSaved(true);invalidate();await refresh();notify('Correction saved at original resolution. Run prediction again to measure agreement with this revision.');
  });
  const position=event=>{const rect=editor.current.getBoundingClientRect();return {x:Math.max(0,Math.min(selected.width,(event.clientX-rect.left)*selected.width/rect.width)),y:Math.max(0,Math.min(selected.height,(event.clientY-rect.top)*selected.height/rect.height))};};
  function paint(a,b){const context=mask.current.getContext('2d');context.strokeStyle=tool==='brush'?'white':'black';context.fillStyle=context.strokeStyle;context.lineWidth=brushSize;context.lineCap='round';context.lineJoin='round';context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();context.beginPath();context.arc(b.x,b.y,brushSize/2,0,Math.PI*2);context.fill();drawEditor();}
  function pointerDown(event){if(!selected||busy||mode==='compare'||event.button!==0)return;event.preventDefault();remember();event.currentTarget.setPointerCapture(event.pointerId);drawing.current=position(event);paint(drawing.current,drawing.current);markDirty();}
  function pointerMove(event){if(!drawing.current)return;const next=position(event);paint(drawing.current,next);drawing.current=next;}
  function endStroke(){if(!drawing.current)return;drawing.current=null;binarize();drawEditor();}
  function historyStep(direction){const from=direction==='undo'?undo:redo,to=direction==='undo'?redo:undo;if(!from.current.length)return;const canvas=mask.current,context=canvas.getContext('2d');to.current.push(context.getImageData(0,0,canvas.width,canvas.height));context.putImageData(from.current.pop(),0,0);updateHistory();markDirty();drawEditor();}
  const viewUrl=value=>value?endpoint(`/predictions/${value.id}/${display}`):null;
  const versionLabel=id=>versions.find(value=>value.id===id)?.label||'Select a model';
  const metrics=prediction?.metrics;
  return <section className="studio-page animate-in"><div className="page-heading-row"><div><div className="eyebrow">{mode==='compare'?'A FAIR, SIDE-BY-SIDE LOOK':'YOUR HUMAN-IN-THE-LOOP WORKSPACE'}</div><h1>{mode==='compare'?'The same road. Different perspectives.':'Every detail deserves a closer look.'}</h1><p>{mode==='compare'?'Compare model versions on the same image, preprocessing, and threshold.':'Inspect a prediction, refine the boundaries, and make your expertise part of the model.'}</p></div><span className="badge"><span className="green-dot"/> LOCAL CPU</span></div>
    <div className="studio-layout"><aside className="image-library panel"><div className="library-title"><h3>Image library</h3><span>{images.length.toString().padStart(2,'0')}</span></div><label className="dropzone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();addFiles([...e.dataTransfer.files]);}}><input ref={upload} id="upload" type="file" accept="image/png,image/jpeg" multiple disabled={!!busy||!project} onChange={e=>{addFiles([...e.target.files]);e.target.value='';}}/><UploadCloud size={25}/><strong>Add road images</strong><span>Drop here or browse files</span><small>PNG / JPEG · Up to 20 MB</small></label><div className="library-images">{!images.length&&<div className="library-empty"><ImagePlus size={23}/><p>A fresh workspace.<br/>Add your first road image.</p></div>}{images.map(row=><button key={row.id} disabled={!!busy} className={`image-item ${selected?.id===row.id?'active':''}`} onClick={()=>run('Opening image…',()=>openImage(row))}><img src={endpoint(`/images/${row.id}/original`)} alt=""/><span><strong>{row.name}</strong><small>{row.width} × {row.height}{row.correction?.complete?' · corrected':''}</small></span>{row.correction?.complete&&<Check size={13}/>}</button>)}</div><div className="library-note"><Layers size={15}/><span>Corrections and versions stay in this project.</span></div></aside>
    <div className="studio-main"><section className="model-controls panel"><label>MODEL VERSION<select id="versionSelect" value={version} disabled={!!busy} onChange={e=>{const chosen=e.target.value;setVersion(chosen);if(versions.find(value=>value.id===chosen)?.phase===9){setThreshold(.5);setInferenceMode('resize');}invalidate();}}>{versions.map(value=><option key={value.id} value={value.id}>{value.label}{value.id===project?.active_version?' · active':''}</option>)}</select></label><label className="range-label">Threshold <output>{threshold.toFixed(2)}</output><input id="threshold" type="range" min="1" max="99" value={Math.round(threshold*100)} disabled={!!busy} onChange={e=>{setThreshold(Number(e.target.value)/100);invalidate();}}/></label><label className="range-label">Overlay <output>{Math.round(opacity*100)}%</output><input id="opacity" type="range" min="0" max="100" value={Math.round(opacity*100)} disabled={!!busy} onChange={e=>{setOpacity(Number(e.target.value)/100);invalidate();}}/></label><button id={mode==='compare'?'compareRun':'predict'} className="btn orange" disabled={!!busy||!selected||!version} onClick={predict}><Play size={14}/>{mode==='compare'?'Run comparison':'Run prediction'}</button></section>
      <InferenceOptions ctx={ctx} version={version} mode={inferenceMode} runtime={runtime} onMode={value=>{setInferenceMode(value);invalidate();}} onRuntime={value=>{setRuntime(value);invalidate();}}/>
      {versions.find(value=>value.id===version)?.phase===9&&<p className="version-note">Multi-source 512 px research model: the official test scored Dice 0.287 overall. Ground camera results were stronger than UAV results. This model can still mark vegetation and miss cracks. The reported score used whole-image resize and threshold 0.5.</p>}
      {mode==='compare'&&<div className="compare-controls"><label>COMPARE AGAINST<select id="compareVersion" value={compareVersion} disabled={!!busy} onChange={e=>{setCompareVersion(e.target.value);setComparison(null);}}>{versions.map(value=><option key={value.id} value={value.id}>{value.label}</option>)}</select></label><p>Accuracy scores appear only for an image with a saved, complete correction.</p></div>}
      <div className="canvas-heading"><div><strong>{selected?.name||'Your next inspection starts here'}</strong><span>{selected?`${selected.width} × ${selected.height} px`:'Add an image from the library.'}</span></div><div><button className="icon-button" aria-label="Zoom out" onClick={()=>setZoom(value=>Math.max(.5,value-.25))}><ZoomOut size={15}/></button><span>{Math.round(zoom*100)}%</span><button className="icon-button" aria-label="Zoom in" onClick={()=>setZoom(value=>Math.min(8,value+.25))}><ZoomIn size={15}/></button><button className="icon-button" aria-label="Fit image" onClick={()=>setZoom(1)}><Expand size={15}/></button><select id="displayMode" aria-label="Prediction display" value={display} onChange={e=>setDisplay(e.target.value)}><option value="overlay">Overlay</option><option value="mask">Binary mask</option><option value="heatmap">Probability</option></select></div></div>
      <div className="viewer-grid"><Viewer title="Original image" tag="ORIGINAL" id="originalViewport" loaded={!!original} image={selected} zoom={zoom} sync={sync} stageRef={originalStage}>{original&&<img src={original.src} alt="Original road image"/>}</Viewer><Viewer title="Model prediction" tag="PREDICTION" id="predictionViewport" loaded={!!prediction} image={selected} zoom={zoom} sync={sync} stageRef={predictionStage}>{prediction&&<img src={viewUrl(prediction)} alt="Model crack prediction"/>}</Viewer><Viewer title={mode==='compare'?'Comparison model':'Your correction'} tag={mode==='compare'?'COMPARISON':saved?'SAVED':isDirty?'UNSAVED':'NOT SAVED'} id="editorViewport" loaded={mode==='compare'?!!comparison:!!original} image={selected} zoom={zoom} sync={sync} stageRef={editorStage} editor>{mode==='compare'?comparison&&<img src={viewUrl(comparison)} alt="Comparison model prediction"/>:<canvas id="editorCanvas" ref={editor} aria-label="Mask correction canvas" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={endStroke} onPointerCancel={endStroke}/>}</Viewer></div>
      <div className="prediction-captions"><span>{selected?'Original resolution preserved':'Ready when you are'}</span><span>{prediction?`${versionLabel(prediction.version_id)} · ${(prediction.latency_ms/1000).toFixed(2)} s`:'No prediction yet'}</span><span id="saveBadge">{mode==='compare'?comparison?`${versionLabel(comparison.version_id)}${comparison.metrics?` · Dice ${(comparison.metrics.dice*100).toFixed(1)}%`:''}`:'Select a comparison model':saved?'SAVED':isDirty?'UNSAVED':'NOT SAVED'}</span></div>
      <div className="studio-metrics"><div><small>CRACK PIXELS</small><strong>{prediction?prediction.crack_pixels.toLocaleString():'—'}</strong><span>At the selected threshold</span></div><div><small>IMAGE COVERAGE</small><strong>{prediction?`${(prediction.area_fraction*100).toFixed(2)}%`:'—'}</strong><span>Fraction of image area</span></div><div><small>DICE / IoU</small><strong id="qualityMetrics">{metrics?`${(metrics.dice*100).toFixed(1)} / ${(metrics.iou*100).toFixed(1)}`:'—'}</strong><span>{metrics?'Against saved correction (%)':'Needs a complete saved correction'}</span></div></div>
      {mode==='inspect'&&<section className="editor-toolbar panel"><div className="tool-group"><button id="brushTool" className={`tool ${tool==='brush'?'selected':''}`} onClick={()=>setTool('brush')} aria-pressed={tool==='brush'}><Brush size={14}/> Brush</button><button id="eraseTool" className={`tool ${tool==='erase'?'selected':''}`} onClick={()=>setTool('erase')} aria-pressed={tool==='erase'}><Eraser size={14}/> Erase</button><label className="brush-size">Size<input id="brushSize" type="range" min="1" max="80" value={brushSize} onChange={e=>setBrushSize(Number(e.target.value))}/><span>{brushSize} px</span></label><button id="undo" className="icon-button" title="Undo" aria-label="Undo" disabled={!history.undo||!!busy} onClick={()=>historyStep('undo')}><Undo2 size={16}/></button><button id="redo" className="icon-button" title="Redo" aria-label="Redo" disabled={!history.redo||!!busy} onClick={()=>historyStep('redo')}><Redo2 size={16}/></button></div><div className="tool-group"><button id="usePrediction" className="text-button" disabled={!prediction||!!busy} onClick={initializeFromPrediction}>Start from prediction</button><button id="aiSuggest" className="text-button" disabled={!selected||!!busy} title={health?.ai_ready?'Find additional crack paths with AI':health?.ai_message} onClick={suggestMissedCracks}>AI find cracks</button><button className="text-button" disabled={!selected||!!busy} onClick={()=>{remember();const context=mask.current.getContext('2d');context.fillStyle='black';context.fillRect(0,0,selected.width,selected.height);markDirty();drawEditor();}}>Clear</button></div><div className="save-group"><label><input id="complete" type="checkbox" checked={complete} disabled={!selected||!!busy} onChange={e=>{setComplete(e.target.checked);markDirty();}}/>Fully corrected</label><button id="saveCorrection" className="btn secondary" disabled={!selected||!!busy} onClick={save}><Save size={14}/> Save correction</button></div><p>AI suggestions are approximate drafts. Review the whole image before marking it complete; drafts cannot enter adaptation.</p></section>}
      <div className="studio-next"><span><Crosshair size={15}/> Pixel measurements need calibration for physical dimensions.</span><button className="text-button" onClick={()=>navigate('adapt')}>Continue to adaptation <ArrowRight size={14}/></button></div>
    </div></div>
  </section>;
}


