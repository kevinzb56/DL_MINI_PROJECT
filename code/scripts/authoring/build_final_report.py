"""Build the final technical report from the consolidated, verified evidence."""
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon
import markdown

ROOT=Path(__file__).resolve().parents[2]
INK='#183238';ORANGE='#bb673c';MUTED='#61767d';W=A4[0]-88


def clean(value):
    return str(value).replace('\u2013','-').replace('\u2014','-').replace('\u2011','-').replace('\u2212','-').replace('\u00b1','+/-').replace('\u2192','to').replace('\u00d7','x')


def figures(evidence,folder):
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.1,3.2),layout='constrained')
    for model,label,color in [('control','LiteCrack',INK),('boundary','LiteCrack-BA',ORANGE)]:
        rows=[r for r in evidence['fewshot'] if r['model']==model]
        ax.errorbar([r['k'] for r in rows],[r['target_dice_percent_mean'] for r in rows],
                    yerr=[r['target_dice_percent_sample_sd'] or 0 for r in rows],marker='o',capsize=4,lw=2,label=label,color=color)
    ax.set(xlabel='Labeled support images',ylabel='Target-query Dice (%)',xticks=[0,1,5,10],ylim=(25,66))
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='lower right')
    fig.savefig(folder/'fewshot.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.1,3.1),layout='constrained')
    x=np.arange(4);models=evidence['static_models']
    ax.bar(x-.17,[100*m['source']['dice'] for m in models],.34,label='Source holdout (237)',color=INK)
    ax.bar(x+.17,[100*m['target']['dice'] for m in models],.34,label='Target query (95)',color=ORANGE)
    ax.set(xticks=x,xticklabels=[m['name'] for m in models],ylabel='Dice (%)',ylim=(0,100));ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
    fig.savefig(folder/'static_comparison.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.7,4.8),layout='constrained')
    for ax,name,title in zip(axes,['wide_resize_overlay.png','wide_tiled_overlay.png'],['Whole-image resize','Overlapping tiles']):
        ax.imshow(PILImage.open(ROOT/'output/phase7'/name));ax.set_title(title);ax.axis('off')
    fig.savefig(folder/'wide_scene_failure.png',dpi=180);plt.close(fig)
    # Native-size actual demo images are exported by the browser workflow.
    import zipfile
    with zipfile.ZipFile(ROOT/'output/phase8/demo/project-export.zip') as z:
        metadata=json.loads(z.read('metadata.json'))
        iid=evidence['demonstration']['source_map']['support']
        pred=next(r for r in metadata['predictions'] if r['image_id']==iid)
        corr=next(r for r in reversed(metadata['corrections']) if r['image_id']==iid and r['complete'])
        import io
        assets=[PILImage.open(io.BytesIO(z.read('images/'+iid+'.png'))),
                PILImage.open(io.BytesIO(z.read('predictions/'+pred['id']+'/overlay.png'))),
                PILImage.open(io.BytesIO(z.read('corrections/'+corr['id']+'.png')))]
        fig,axes=plt.subplots(1,3,figsize=(8.1,3.5),layout='constrained')
        for ax,image,title in zip(axes,assets,['Real acquisition image','Actual trained prediction','Existing reference mask']):
            ax.imshow(image,cmap='gray' if image.mode=='L' else None);ax.set_title(title,fontsize=9);ax.axis('off')
        fig.savefig(folder/'real_demo.png',dpi=200);plt.close(fig)


def build():
    evidence=json.loads((ROOT/'output/phase8/evidence.json').read_text())
    folder=ROOT/'output/phase8/figures';figures(evidence,folder)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9.5,leading=13.5,textColor=colors.HexColor(INK),spaceAfter=9))
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=colors.HexColor(INK),spaceAfter=17))
    styles.add(ParagraphStyle(name='ReportKicker',fontName='Helvetica-Bold',fontSize=8,leading=11,textColor=colors.HexColor(ORANGE),spaceAfter=9))
    styles.add(ParagraphStyle(name='ReportSmall',fontName='Helvetica',fontSize=8,leading=11,textColor=colors.HexColor(MUTED),spaceAfter=9))
    styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8,leading=10.5,textColor=colors.HexColor(INK)))
    styles.add(ParagraphStyle(name='HeadCell',parent=styles['Cell'],fontName='Helvetica-Bold',textColor=colors.white))
    story=[];md=['# CrackLab: final technical report','Release 1.0.0-pilot | 1 November 2026','']
    def p(text,style='ReportBody'):
        story.append(Paragraph(html.escape(clean(text)).replace('\n','<br/>'),styles[style]));md.extend([clean(text),''])
    def page(title,kicker):
        if story:story.append(PageBreak())
        p(kicker,'ReportKicker');p(title,'ReportTitle');md.extend(['## '+title,''])
    def table(headers,rows,widths=None):
        values=[[Paragraph(html.escape(clean(v)),styles['HeadCell']) for v in headers]]
        values.extend([Paragraph(html.escape(clean(v)).replace('\n','<br/>'),styles['Cell']) for v in row] for row in rows)
        result=Table(values,colWidths=widths or [W/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(INK)),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#d8e1e1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f4')])]))
        story.extend([result,Spacer(1,12)])
        md.append('| '+' | '.join(clean(v).replace('\n',' ') for v in headers)+' |');md.append('| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(clean(v).replace('\n',' ') for v in row)+' |' for row in rows);md.append('')
    def figure(name,caption,maxheight=260):
        path=folder/name;image=PILImage.open(path);width,height=image.size;scale=min(W/width,maxheight/height)
        picture=Image(str(path),width=width*scale,height=height*scale);picture.hAlign='CENTER'
        story.extend([picture,Spacer(1,7),Paragraph(html.escape(clean(caption)),styles['ReportSmall'])])
        md.extend([f'![{clean(caption)}](../output/phase8/figures/{name})',''])
    def score(value):return f'{100*value:.2f}%'
    def fs(model,k):return next(r for r in evidence['fewshot'] if r['model']==model and r['k']==k)

    page('Lightweight road crack segmentation with boundary-aware few-shot adaptation','CRACKLAB / FINAL TECHNICAL REPORT / 1.0.0-PILOT')
    p('A reproducible research pilot and a local inspection application.','ReportTitle')
    p('Release date: 1 November 2026. Prepared from the preserved experiment archives, current application source, clean-environment tests and recorded demonstration. No institutional approval or authorship fields are inferred.','ReportSmall')
    p('Abstract','ReportKicker')
    p('This project extends a paper-inspired road-crack segmentation baseline with a compact MobileNetV3-Small encoder, a separable decoder, auxiliary boundary supervision and bounded decoder-only adaptation. A frozen DeepCrack-to-CrackMap pilot separates source training, source validation, source holdout, target acquisition and target query. Thirty fixed-budget adaptations assess 1/5/10-image support sets across five draws for two parents.')
    p('The region-only network has 339,265 parameters, 83.92% fewer than the implemented R2AU baseline. Static target Dice increases from 34.01% to 41.72% with boundary supervision. Boundary-aware 10-shot mean target Dice reaches 59.09%, while source-holdout Dice falls by 4.87 points relative to its static parent. These are single-seed exploratory results on a previously inspected query set, not an exact paper reproduction or independent confirmation.')
    p('The delivered React/FastAPI application supports inspection, pixel-aligned correction, paired comparison, recoverable adaptation, retention validation, entropy/random suggestions, blended tiled inference and verified ONNX inference. A fresh release virtual environment passed 49 CPU tests with six CUDA-only skips. A real-acquisition-image application workflow is recorded; its reference masks are existing annotations, not measured expert corrections.')
    table(['Release outcome','Scientific boundary'],[['Functional pilot delivered','Multi-seed confirmation and full GPU retraining remain unperformed.'],['Integrity-checked application bundle','Failed retention gates and broad-scene false positives remain explicit.']],[W*.43,W*.57])

    page('Data, split controls and evaluation protocol','01 / EXPERIMENTAL FOUNDATION')
    p('DeepCrack is the source dataset and CrackMap is the target. The local source is narrower than the reference paper\'s five-source protocol. The frozen manifest contains 657 pairs and is identified by SHA256 '+evidence['manifest_sha256']+'.','ReportSmall')
    table(['Partition','Images','Permitted use'],[[k.replace('_',' '),v,{'source_train':'Optimizer updates and optional replay','source_validation':'Validation and fixed retention subset','source_holdout':'Fixed source evaluation only','target_acquisition':'Support labels and local demonstration','target_query':'Fixed query evaluation; never support selection'}[k]] for k,v in evidence['split_counts'].items()],[W*.31,W*.12,W*.57])
    p('Input preprocessing uses RGB divided by 255 and bilinear square resize to 256 x 256. Binary masks use nearest-neighbor resize during training and 0/255 storage. The fixed evaluation threshold is 0.50. UI thresholds are display/inference settings and do not rewrite the benchmark.')
    p('Results are per-image macro means. Dice and IoU equal one for two empty masks and zero for a one-sided empty mask; both-empty cases must be interpreted separately. Boundary F1 uses a fixed two-pixel tolerance at evaluation resolution. Acquisition/query separation, checksum checks and group proxies are preserved.')
    p('Grouping uses filename captures and image-similarity proxies; physical site/sequence independence is unverified. The target query was inspected in earlier phases. Some immutable source-validation IDs contain source_train because their names encode the native dataset partition; the frozen split field determines actual membership.')
    p('Only the 24 replay/source-validation pairs and fixed acquisition assets needed by later checks were restored locally. Archived evidence is not presented as a new full-image re-audit of all 657 pairs.')

    page('Model and application implementation','02 / SYSTEM DESIGN')
    p('U-Net and a paper-inspired R2AU implementation establish fixed references. LiteCrack uses a half-width MobileNetV3-Small encoder initialized without pretrained weights, GroupNorm, multiscale skip connections and depthwise-separable decoder blocks. LiteCrack-BA adds a region-sharing boundary head; it does not guarantee topological connectivity.')
    table(['Model','Parameters','Role'],[[m['name'],f'{m["parameters"]:,}','Frozen baseline' if i<2 else 'Region-only control' if i==2 else 'Boundary-aware treatment'] for i,m in enumerate(evidence['static_models'])],[W*.4,W*.22,W*.38])
    drawing=Drawing(W,105)
    for index,(label,sub) in enumerate([('RGB image','256 x 256'),('Compact encoder','MobileNetV3-Small'),('Separable decoder','Skip connections'),('Output heads','Region + boundary')]):
        x=index*(W/4);drawing.add(Rect(x,35,W/4-10,58,rx=5,fillColor=colors.HexColor('#eef3f2'),strokeColor=colors.HexColor('#c5d4d4')))
        drawing.add(String(x+9,71,label,fontName='Helvetica-Bold',fontSize=8,fillColor=colors.HexColor(INK)))
        drawing.add(String(x+9,53,sub,fontName='Helvetica',fontSize=7,fillColor=colors.HexColor(MUTED)))
        if index<3:drawing.add(Line(x+W/4-8,64,x+W/4-1,64,strokeColor=colors.HexColor(ORANGE)))
    story.append(drawing)
    p('Training uses stabilized BCE+Dice, GroupNorm, clipping and validation-based checkpoint selection. Boundary targets are a 3 x 3 morphological gradient, with the fixed auxiliary loss added to the region objective. Adaptation freezes encoder parameters and updates decoder/output heads with Adam 0.0001, batch two and a predeclared support-pass budget.')
    p('The application separates React state and original-resolution canvas editing from FastAPI inference, SQLite metadata and filesystem snapshots. Workers run in isolated processes. Every complete optimizer step saves immutable model/optimizer/RNG/sampler state with checksummed pointers. Draft corrections cannot become fully labeled support; activation remains an explicit action after validation.')
    p('Phase 7 was added without changing the Phase 6 worker or engine, preserving existing recovery code identities. The package includes trusted trained archives; browser users cannot upload executable checkpoint files or arbitrary server paths.')

    page('Static comparison and the domain gap','03 / FROZEN CHECKPOINT RESULTS')
    table(['Model','Source Dice','Target Dice','Target IoU'],[[m['name'],score(m['source']['dice']),score(m['target']['dice']),score(m['target']['iou'])] for m in evidence['static_models']],[W*.37,W*.21,W*.21,W*.21])
    figure('static_comparison.png','Figure 1. Same frozen source holdout (237) and target query (95), macro means, threshold 0.50. Single training seed; no universal superiority claim.',235)
    p('Region-only LiteCrack is 0.51 target Dice point below R2AU with much lower parameter count, but source Dice is 5.06 points lower. Boundary supervision increases target Dice by 7.71 points and target boundary F1 by 6.22 points. Source Dice and source boundary F1 decrease slightly, so the boundary benefit is target-specific.')
    p('The initial Dice-only baselines collapsed and were retained as failure evidence. Stabilized runs are reported separately. Later training configurations were not selected from target-query labels. Earlier query inspection nevertheless limits the final experiment\'s confirmatory status.')

    page('Few-shot adaptation across five support draws','04 / FIXED-BUDGET ADAPTATION')
    rows=[]
    for k in (0,1,5,10):
        a,b=fs('control',k),fs('boundary',k)
        fmt=lambda r:f'{r["target_dice_percent_mean"]:.2f}'+(f' +/- {r["target_dice_percent_sample_sd"]:.2f}' if k else '')
        rows.append([str(k),fmt(a),fmt(b),'Reference' if not k else f'{b["source_dice_change_points_mean"]:+.2f} pts'])
    table(['Labels','LiteCrack Dice (%)','LiteCrack-BA Dice (%)','BA source change'],rows,[W*.1,W*.3,W*.32,W*.28])
    figure('fewshot.png','Figure 2. Means across five fixed support draws; error bars are sample SD across draws, not population confidence intervals. Every draw uses the same query images.',235)
    p('There are 30 independent adaptation initializations: two parent models x five draws x K=1/5/10. Each starts from its original parent; K=10 does not continue from a K=5 candidate. Five support passes give different step counts as K changes; this is a fixed-pass, not equal-compute, comparison.')
    p('Boundary-aware five-shot mean gain is 15.51 target Dice points over its own static model; ten-shot gain is 17.38 points. Source-holdout changes of -2.81 and -4.87 points fail the optional two-point retention target. Existing public masks were used; actual correction minutes were not measured.')

    page('Descriptive uncertainty and its limits','05 / RETROSPECTIVE PAIRED ANALYSIS')
    names={'boundary_minus_region_source_holdout':'BA - region, source','boundary_minus_region_target_query':'BA - region, target',
           'boundary_k5_minus_static_target_query':'BA 5-shot gain, target','boundary_k10_minus_static_target_query':'BA 10-shot gain, target',
           'boundary_k5_minus_static_source_holdout':'BA 5-shot change, source','boundary_k10_minus_static_source_holdout':'BA 10-shot change, source'}
    table(['Comparison','Change (pts)','95% interval','Groups'],[[names[key],f'{r["difference_points"]:+.2f}',f'[{r["lower_95"]:+.2f}, {r["upper_95"]:+.2f}]',r['proxy_groups']] for key,r in evidence['descriptive_intervals'].items()],[W*.4,W*.16,W*.29,W*.15])
    p('Method: 10,000 paired proxy-group bootstrap resamples, seed 881, percentile intervals and an image-weighted mean. Group labels come from the frozen manifest. For adaptation, scores are averaged across the five support draws for each image before resampling; repeated draws are not treated as independent new test images.')
    p('These intervals are a retrospective descriptive analysis of the same previously inspected images. Proxy capture/similarity groups are not verified physical road/site independence. Intervals are unadjusted across the listed comparisons. They are not formal confirmatory significance tests, between-training-seed intervals, or evidence of broad generalization.')
    p('The project still lacks repeated training seeds for the main static comparison and an independently reserved second target domain. The narrow conclusions are compactness, target-specific boundary/adaptation gains on this pilot, and a measurable source-domain tradeoff.')

    page('Extended release: replay, tiles and portability','06 / PHASE 7')
    p('Entropy/random ranking only sees eligible incomplete acquisition images; it excludes complete corrections, frozen evaluation matches and existing job validation images. Entropy is not calibrated confidence. Supports are still chosen manually. Tiling retains spatial scale with 256-pixel inputs, 64-pixel overlap, edge padding and probability blending.')
    paired=evidence['phase7']['paired_source_validation'];jobs=evidence['phase7']['jobs']
    table(['Parent / condition','Local target Dice','Source subset change','Replay gate'],[[('LiteCrack' if index<2 else 'LiteCrack-BA')+(' / replay' if j['settings'].get('source_replay') else ' / no replay'),score(j['validation']['candidate_dice']),f'{paired[index]["source_change_points"]:+.2f} pts','Rejected' if j['settings'].get('source_replay') else 'Source gate not applied'] for index,j in enumerate(jobs)],[W*.36,W*.2,W*.24,W*.2])
    p('This paired smoke pilot uses draw 0 K=5, one support pass, five separate acquisition-validation images and eight fixed source-validation images. Replay mixes one source-training image per support image. It reduced forgetting, but source-subset losses still exceeded two points. Both replay candidates remain rejected. This is not a full E7 or source-holdout confirmation.')
    table(['Model','Native CPU p95','ONNX maximum error'],[[m['name'],f'{b["p95_ms"]:.2f} ms',f'{e["max_probability_difference"]:.8f}'] for m,b,e in zip(evidence['static_models'][2:],evidence['phase7']['benchmark'],evidence['phase7']['onnx_exports'])],[W*.4,W*.3,W*.3])
    p('CPU timing: i5-1335U, PyTorch 2.15.0+cpu, FP32, 256 x 256, batch one, two threads, 20 warmups and 200 timed model passes. Loading/preprocessing/rendering are excluded. ONNX FP32 opset 17 passes max probability error <=0.001 and per-image Dice/IoU difference <=0.002 on eight audited source-validation images; unlabelled and synthetic diagnostics are checked too.')

    page('Recorded real-image application demonstration','07 / FUNCTIONAL EVIDENCE')
    figure('real_demo.png','Figure 3. Real known acquisition image, actual frozen-checkpoint prediction and existing dataset reference mask. The reference was imported explicitly; no measured expert correction or independent evaluation is claimed.',260)
    p('The demonstration runs against a fresh extracted release and separate virtual environment. It covers upload, native prediction, actual brush/undo/redo and a draft save; reference-mask import and agreement metrics; tiled and ONNX inference; entropy ranking; a real source-replay subprocess; candidate validation; paired comparison; project/recovery ZIPs; and reload persistence.')
    demo=evidence['demonstration'];job=demo['job']
    table(['Observed step','Result'],[['Real images','Two fixed target-acquisition records, with hashes and split provenance.'],['Adaptation job',job['state']+'; target/source validation reviewed without changing gates.'],['Candidate action','Activated then rolled back' if demo['candidate_activated_and_rolled_back'] else 'Rejected/not activated according to the recorded validation gate.'],['Exports','Project ZIP, recovery checkpoint SHA256 and ONNX graph checksum verified.'],['Browser errors',str(len(demo['javascript_errors']))+' JavaScript errors; recorded walkthrough.webm retained.']],[W*.26,W*.74])
    p('Full-resolution screenshots, the video, source annotation provenance, per-image records and validation details are under output/phase8/demo/. This demonstrates application functionality on real images, not a field deployment, user study or fresh accuracy benchmark.','ReportSmall')

    page('Acceptance ledger and clean installation','08 / RELEASE VERIFICATION')
    table(['Gate','Status','Finding'],[[r['id'],r['status'].replace('_',' '),r['finding']] for r in evidence['gates']],[W*.08,W*.26,W*.66])
    p('Fresh Python 3.12 virtual environment: 49 passed, six CUDA-only skips. Installed distributions match the release lock. The extracted app imports all 32 references and bundled replay pairs without a dataset download or Node rebuild. Application code and compiled frontend fingerprints match the tested candidate.','ReportSmall')

    page('Known broad-scene failure and limitations','09 / WHAT IS NOT SOLVED')
    figure('wide_scene_failure.png','Figure 4. The same unlabelled wide road photograph in resize and tiled modes. Green denotes predictions, not ground truth; trees, sky edges or lamp regions still generate false positives.',330)
    p('All 32 original base/adapted references misclassified substantial vegetation on this photograph. Tiling changes the pattern and recovers additional foreground cracks but does not become a road-region detector. No Dice/IoU or overall winner is assigned without a complete ground-truth mask.')
    p('Remaining limits include single-seed training, query reuse, incomplete physical grouping, the absence of a second independent target, no measured annotation effort, failed retention targets, and source/target distribution shifts. No output establishes road safety, crack depth, physical dimensions or maintenance priority.')
    p('DeepCrack source terms restrict the data to noncommercial research/education. CrackMap usage terms remain unconfirmed in the frozen protocol; this local package does not assert public redistribution permission. Third-party data and the reference paper are not relicensed by the repository code license.','ReportSmall')

    page('Reproduction, provenance and final scope','10 / HANDOFF')
    p('1. Extract the versioned bundle into a new folder. Run scripts/verify_release.py before using trusted checkpoint files. 2. Run Setup_CrackLab.ps1 to create a Python 3.12 virtual environment and install pinned runtime dependencies. 3. Start_CrackLab.cmd serves the compiled interface at 127.0.0.1:8765. 4. Use the included acquisition demo only with its recorded annotation provenance. 5. Keep original archives and recovery backups; never overwrite a newer workspace.')
    p('The release includes Phase 3/4/5 trained archives, the compiled React app, source, tests, model card, setup/demo guides, selected verified replay/demo assets and Phase 7 recovery evidence. It excludes the main user workspace, credentials, venv, node_modules and browser binaries. Network access is needed for the initial dependency installation; no offline wheelhouse is claimed.')
    p('Full original GPU training has not been rerun from this CPU release. Notebook/configuration commands and frozen manifests remain available, but require the original compatible GPU environment and full permitted datasets. Code, weights, application reproducibility and scientific generalization are separate evidence levels.')
    table(['Reference','Source'],[['[1] Katsamenis et al. (2023)','A Few-Shot Attention Recurrent Residual U-Net for Crack Segmentation. arXiv:2303.01582.'],['[2] Liu et al. (2019)','DeepCrack. Neurocomputing 338:139-153. doi:10.1016/j.neucom.2019.01.036.'],['[3] Pinned datasets','github.com/yhlleo/DeepCrack; github.com/ikatsamenis/CrackMap. Commits are retained in configs/data_sources.json.'],['[4] Runtime APIs','docs.pytorch.org/docs/stable/onnx/; onnxruntime.ai/docs/api/python/api_summary.html.'],['[5] Local evidence','output/phase8/evidence.json; Phase 5 archive audit; Phase 7 parity/pilot reports; fresh tests and recorded demo.']],[W*.35,W*.65])
    p('Final conclusion: the pilot application and versioned research package are delivered with reproducible model identities and explicit scope. Target-specific gains and compactness are supported by preserved evidence; universal generalization, full retention and independent scientific confirmation remain future work.')

    def decorate(canvas,doc):
        canvas.saveState();width,height=A4
        canvas.setStrokeColor(colors.HexColor(ORANGE));canvas.setLineWidth(1.4);canvas.line(44,height-34,width-44,height-34)
        canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor(MUTED));canvas.drawString(44,height-25,'CRACKLAB / FINAL TECHNICAL REPORT')
        canvas.drawString(44,25,'1.0.0-pilot | 1 November 2026 | Research / education prototype')
        canvas.drawRightString(width-44,25,str(doc.page));canvas.restoreState()
    pdf=ROOT/'output/pdf/CrackLab_Final_Report.pdf';pdf.parent.mkdir(exist_ok=True)
    document=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=53,bottomMargin=44,title='CrackLab final technical report',author='CrackLab project')
    document.build(story,onFirstPage=decorate,onLaterPages=decorate)
    text='\n'.join(md)+'\n';(ROOT/'docs/FINAL_REPORT.md').write_text(text,encoding='utf-8')
    body=markdown.markdown(text,extensions=['tables'])
    page_html='<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="../../docs/"><title>CrackLab final report</title><style>body{max-width:950px;margin:45px auto;padding:0 28px;font:16px/1.7 Segoe UI,Arial;color:#183238}h1,h2{line-height:1.3}h2{margin-top:45px;border-bottom:2px solid #bb673c;padding-bottom:10px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border:1px solid #d8e1e1;text-align:left}th{background:#183238;color:white}img{max-width:100%}p{margin:13px 0}</style></head><body>'+body+'</body></html>'
    (ROOT/'output/phase8/Final_Report.html').write_text(page_html,encoding='utf-8')
    print('Built PDF, Markdown and HTML report from one evidence source.')


if __name__=='__main__':build()
