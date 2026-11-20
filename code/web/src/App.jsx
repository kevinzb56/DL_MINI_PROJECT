import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowUpRight, Check, ChevronDown, Cpu, Download, FlaskConical, GitBranch, LayoutDashboard, Menu, Plus, ScanLine, ShieldCheck, X, Zap } from 'lucide-react';
import { api, post } from './api';
import Overview, { Research } from './Overview';
import Studio from './Studio';
import Adaptation from './Adaptation';

const pages = { overview: 'Project overview', studio: 'Inspection studio', compare: 'Model comparison', adapt: 'Adaptation lab', research: 'Research & results' };
const navigation = [{id:'overview',icon:LayoutDashboard,label:'Overview'},{id:'studio',icon:ScanLine,label:'Inspection studio'},{id:'compare',icon:Activity,label:'Compare models'},{id:'adapt',icon:GitBranch,label:'Adaptation lab'},{id:'research',icon:FlaskConical,label:'Research & results'}];
const initialPage = () => pages[location.hash.slice(1)] ? location.hash.slice(1) : 'overview';

export default function App() {
  const [page, setPage] = useState(initialPage);
  const [projects, setProjects] = useState([]), [project, setProject] = useState(null);
  const [images, setImages] = useState([]), [versions, setVersions] = useState([]), [jobs, setJobs] = useState([]);
  const [busy, setBusy] = useState(''), [message, setMessage] = useState(null), [health, setHealth] = useState(null);
  const [dialog, setDialog] = useState(false), [name, setName] = useState(''), [mobileMenu, setMobileMenu] = useState(false);
  const [dirty, setDirty] = useState(false);
  const lock = useRef(false), projectRef = useRef(null);
  const endpoint = useCallback(path => `/api/projects/${projectRef.current?.id}${path}`, []);
  const notify = useCallback((text, error=false) => setMessage({text,error}), []);
  const run = useCallback(async (label, fn) => {
    if (lock.current) return;
    lock.current = true; setBusy(label);
    try { return await fn(); } catch (error) { notify(error.message,true); }
    finally { lock.current = false; setBusy(''); }
  }, [notify]);
  const refresh = useCallback(async () => {
    const pid = projectRef.current?.id;
    if (!pid) return;
    const [nextImages, nextVersions, nextJobs] = await Promise.all([api(endpoint('/images')),api(endpoint('/versions')),api(endpoint('/jobs'))]);
    if (projectRef.current?.id !== pid) return;
    setImages(nextImages); setVersions(nextVersions); setJobs(nextJobs);
  }, [endpoint]);
  const openProject = useCallback(async pid => {
    const next = await api(`/api/projects/${pid}`);
    projectRef.current = next; setProject(next); setDirty(false);
    localStorage.setItem('cracklab-project',pid);
    await refresh();
  }, [refresh]);
  const navigate = useCallback(next => {setPage(next);location.hash=next;setMobileMenu(false);window.scrollTo({top:0,behavior:'smooth'});}, []);
  useEffect(() => {const onHash=()=>setPage(initialPage());window.addEventListener('hashchange',onHash);return()=>window.removeEventListener('hashchange',onHash);},[]);
  useEffect(() => {run('Opening your workspace…',async()=>{
    setHealth(await api('/api/health'));
    let list=await api('/api/projects');
    if(!list.length)list=[await api('/api/projects',post({name:'Road inspection'}))];
    setProjects(list);
    const saved=localStorage.getItem('cracklab-project');
    await openProject((list.find(p=>p.id===saved)||list[0]).id);
  });},[openProject,run]);
  useEffect(()=>{if(!message||message.error)return;const id=setTimeout(()=>setMessage(null),8500);return()=>clearTimeout(id);},[message]);
  useEffect(()=>{const handler=event=>{if(dirty){event.preventDefault();event.returnValue='';}};window.addEventListener('beforeunload',handler);return()=>window.removeEventListener('beforeunload',handler);},[dirty]);
  useEffect(()=>{
    if(!jobs.some(job=>['queued','running'].includes(job.state)))return;
    const id=setInterval(()=>{if(!lock.current)refresh().catch(error=>notify(error.message,true));},2000);
    return()=>clearInterval(id);
  },[jobs,refresh,notify]);
  const discard = () => !dirty || window.confirm('Discard unsaved correction changes?');
  const activate = async versionId => {
    const next=await api(endpoint('/activate'),post({version_id:versionId}));
    projectRef.current=next;setProject(next);await refresh();notify('Model activated. Earlier versions remain available for rollback.');
  };
  const ctx = {project,images,versions,jobs,busy,endpoint,run,notify,refresh,navigate,activate,setDirty,dirty,health};
  return <div className="app-shell">
    <aside className={`sidebar ${mobileMenu?'mobile-open':''}`}>
      <a href="#overview" className="brand" onClick={()=>navigate('overview')}><span className="brand-mark"><Zap size={24} strokeWidth={1.7}/></span><span>cracklab<span className="brand-period">.</span><small>ROAD VISION STUDIO</small></span></a>
      <div className="workspace-select"><div><span className="workspace-avatar">R</span><label className="sr-only" htmlFor="projectSelect">Inspection project</label><select id="projectSelect" value={project?.id||''} disabled={!!busy} onChange={e=>{if(discard())run('Opening project…',()=>openProject(e.target.value));}}>{projects.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</select><ChevronDown size={13}/></div><button id="newProject" title="New project" aria-label="New project" onClick={()=>{setName('');setDialog(true);}}><Plus size={15}/></button></div>
      <div className="nav-label">WORKSPACE</div><nav>{navigation.map(({id,icon:Icon,label})=><button key={id} data-tab={id} className={`nav-item ${page===id?'active':''}`} onClick={()=>navigate(id)}><Icon size={17}/><span>{label}</span>{id==='studio'&&<small>LIVE</small>}{page===id&&<i/>}</button>)}</nav>
      <div className="sidebar-research"><span className="research-label"><FlaskConical size={15}/> THE PROJECT</span><p>Lightweight by design.<br/>Boundary-aware by learning.</p><span className="research-pill">LiteCrack-BA <span>v1.0</span></span></div>
      <div className="sidebar-bottom"><div><span className="green-dot"/><span id="connection">{health?`${health.models} models ready`:'Backend offline'}</span><Cpu size={14}/></div><p>Runs on your machine.<br/>Your images stay with you.</p><span className="sidebar-divider"/><div className="author"><span>KS</span><div><strong>Research workspace</strong><small>Local · Single-user</small></div><ShieldCheck size={15}/></div></div>
    </aside>
    <div className="app-main"><header className="topbar"><div><button className="mobile-menu icon-button" aria-label="Toggle navigation" onClick={()=>setMobileMenu(!mobileMenu)}><Menu size={19}/></button><span className="breadcrumb-root">Workspace</span><span className="breadcrumb-slash">/</span><strong>{pages[page]}</strong></div><div className="header-right"><span className="local-badge"><i className="green-dot"/> LOCAL SESSION</span>{project&&<a id="export" className="btn secondary" href={endpoint('/export')} onClick={e=>{if(dirty){e.preventDefault();notify('Save your correction before exporting.',true);}}}><Download size={14}/> Export project</a>}</div></header>
      {message&&<div className={`notification ${message.error?'error':''}`} role={message.error?'alert':'status'}><span>{message.text}</span><button aria-label="Dismiss message" onClick={()=>setMessage(null)}><X size={16}/></button></div>}
      <main className="page-content"><div hidden={page!=='overview'}><Overview navigate={navigate} models={health?.models}/></div><div hidden={!['studio','compare'].includes(page)}><Studio key={project?.id||'none'} ctx={ctx} mode={page==='compare'?'compare':'inspect'}/></div><div hidden={page!=='adapt'}><Adaptation ctx={ctx}/></div><div hidden={page!=='research'}><Research/></div></main>
      <footer className="app-footer"><span><Zap size={12}/> CRACKLAB <i>/</i> A closer look at every road.</span><span>Research prototype <span className="footer-dot">·</span> Built with React & PyTorch</span></footer>
    </div>
    <div id="busy" className="busy-toast" role="status" hidden={!busy}><span className="spinner"/>{busy}</div>
    {dialog&&<div className="modal-backdrop" onClick={()=>setDialog(false)}><section className="project-modal" role="dialog" aria-modal="true" aria-labelledby="dialogTitle" onClick={e=>e.stopPropagation()}><button className="modal-close icon-button" aria-label="Close new project" onClick={()=>setDialog(false)}><X size={19}/></button><div className="modal-icon"><Plus size={25}/></div><div className="eyebrow">A NEW PERSPECTIVE</div><h2 id="dialogTitle">Start an inspection project.</h2><p>Keep your images, corrections, and model history together.</p><form id="projectForm" onSubmit={e=>{e.preventDefault();if(!discard())return;run('Creating project…',async()=>{const value=await api('/api/projects',post({name}));setProjects(list=>[...list,value]);await openProject(value.id);setDialog(false);navigate('studio');});}}><label>Project name<input autoFocus id="projectName" value={name} onChange={e=>setName(e.target.value)} required maxLength={100} placeholder="e.g. Campus access road"/></label><button className="btn orange" disabled={!!busy} type="submit">Create project <ArrowUpRight size={16}/></button></form></section></div>}
  </div>;
}
