from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'presentations' / 'CrackLab_Five_Slide_Research_Presentation.pptx'
BG = ROOT / 'output' / 'presentations' / 'template_background.png'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def box(slide, x, y, w, h, fill=(255,255,255), radius=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                    Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor(*fill)
    shape.line.fill.background()
    return shape

def txt(slide, text, x, y, w, h, size=20, bold=False, color=(25,25,25), font='Arial', align=None):
    s=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=s.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.02); tf.margin_top=tf.margin_bottom=Inches(.02)
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line; p.font.name=font; p.font.size=Pt(size); p.font.bold=bold
        p.font.color.rgb=RGBColor(*color); p.space_after=Pt(11)
        if align is not None:p.alignment=align
    return s

def slide(title, number):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(BG),0,0,prs.slide_width,prs.slide_height)
    box(s,.55,.38,12.22,6.73,(247,247,247))
    box(s,.72,.55,11.9,6.36,(255,255,255))
    txt(s,title,1.1,.76,11.1,.63,34,True,(15,15,15),'Arial Narrow')
    box(s,1.08,1.48,11.15,.025,(95,102,115))
    txt(s,f'CrackLab  /  {number:02d}',10.75,6.55,1.45,.25,10,False,(105,105,105))
    return s

s=slide('Problem Statement',1)
txt(s,'Road cracks need pixel-level detection',1.12,1.83,10.8,.55,25,True)
txt(s,'Manual inspection is slow and visual interpretation varies across reviewers. A useful model must locate thin cracks in an image, not only classify the whole image.',1.12,2.5,10.8,1.05,22)
box(s,1.1,3.9,11.1,1.88,(241,242,244),True)
txt(s,'Core research challenge',1.42,4.18,9.9,.43,22,True)
txt(s,'Accuracy drops when the camera, road surface, lighting, and capture height change. Predictions also need a human review path before use in inspection.',1.42,4.68,9.8,.78,20)
s.notes_slide.notes_text_frame.text='Sources: docs/FINAL_REPORT.md; MODEL_CARD.md. This is a research and human-inspection system, not a road-safety assessment.'

s=slide('Proposed System',2)
txt(s,'Image  →  crack mask  →  human correction',1.12,1.85,10.8,.6,25,True)
txt(s,'LiteCrack uses a compact MobileNetV3-Small encoder and a multiscale decoder to segment crack pixels. LiteCrack-BA adds boundary supervision to better trace crack edges.',1.12,2.55,10.9,1.2,21)
box(s,1.1,4.0,5.25,1.66,(241,242,244),True)
box(s,6.58,4.0,5.63,1.66,(241,242,244),True)
txt(s,'Model',1.4,4.28,4.6,.35,21,True)
txt(s,'RGB image → probability mask\n256 px inference tiles in the app',1.4,4.76,4.65,.75,17)
txt(s,'CrackLab application',6.88,4.28,4.9,.35,21,True)
txt(s,'Review, brush correction, compare and export masks',6.88,4.76,4.9,.75,17)
s.notes_slide.notes_text_frame.text='Sources: MODEL_CARD.md; docs/FINAL_REPORT.md; docs/MULTISOURCE_APP_DEMO.md. App inference differs from benchmark preprocessing.'

s=slide('Dataset Used',3)
txt(s,'Primary experiment: DeepCrack → CrackMap',1.12,1.82,10.8,.53,24,True)
txt(s,'Frozen image–mask pairs',1.12,2.44,4.8,.4,19,True)
txt(s,'240  source training\n60    source validation\n237  source holdout\n25    target acquisition\n95    target query',1.12,2.92,4.72,2.8,20)
box(s,6.25,2.3,5.92,3.65,(241,242,244),True)
txt(s,'Separate multi-source follow-up',6.55,2.62,5.15,.55,21,True)
txt(s,'Six audited archives: ground GoPro, ground iPhone, and four UAV groups.\n\n2,526 train  •  542 validation\n496 publisher test images',6.55,3.35,5.2,2.12,19)
txt(s,'The two experiments use different splits; their Dice scores are not directly comparable.',1.12,6.03,10.7,.38,15,False,(80,80,80))
s.notes_slide.notes_text_frame.text='Sources: MODEL_CARD.md, data and protocol; output/multisource_test_v1/FINAL_TEST_REPORT.md; output/multisource_full_v1/TRAINING_REPORT.md.'

s=slide('Result Analysis',4)
txt(s,'Primary target query: boundary supervision helps',1.12,1.8,10.8,.55,23,True)
box(s,1.1,2.52,5.25,2.4,(241,242,244),True)
txt(s,'34.01%',1.42,2.82,4.55,.85,36,True,(84,93,112))
txt(s,'LiteCrack target Dice',1.42,3.82,4.5,.4,20)
box(s,6.57,2.52,5.62,2.4,(241,242,244),True)
txt(s,'41.72%',6.9,2.82,4.8,.85,36,True,(84,93,112))
txt(s,'LiteCrack-BA target Dice',6.9,3.82,4.8,.4,20)
txt(s,'Separate six-source test: mean Dice 28.73% across 496 images. Ground-camera Dice was 69–72%; UAV groups were 14–22%.',1.12,5.27,10.9,.94,19)
txt(s,'Interpretation: useful research candidate with human review; distant UAV cracks remain a major weakness.',1.12,6.22,10.8,.4,15,False,(80,80,80))
s.notes_slide.notes_text_frame.text='Sources: MODEL_CARD.md, static results; output/multisource_test_v1/FINAL_TEST_REPORT.md. Primary target query and separate publisher source-holdout use different datasets and protocols. Six-source values are rounded from 0.287325, 0.693625/0.719845, and 0.142357–0.216736.'

s=slide('Future Scope',5)
items=[('1','Improve UAV performance','Add complete masks and difficult negative examples from varied heights; test by capture group.'),('2','Validate generalization','Evaluate new physical sites and camera conditions without tuning on the frozen test split.'),('3','Strengthen inspection workflow','Measure reviewer correction time and error rates before considering wider field use.')]
for i,(n,heading,body) in enumerate(items):
    y=1.9+i*1.45
    box(s,1.1,y,.68,.68,(84,93,112),True)
    txt(s,n,1.1,y+.08,.68,.45,22,True,(255,255,255),align=PP_ALIGN.CENTER)
    txt(s,heading,2.04,y-.02,9.7,.44,22,True)
    txt(s,body,2.04,y+.51,9.7,.66,17)
s.notes_slide.notes_text_frame.text='Future work inferred from MODEL_CARD.md limitations, output/multisource_test_v1/FINAL_TEST_REPORT.md, and docs/FINAL_REPORT.md. These are proposed steps, not completed results.'

OUT.parent.mkdir(parents=True,exist_ok=True)
prs.save(OUT)
print(OUT)
