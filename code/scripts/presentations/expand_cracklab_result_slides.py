"""Expand one CrackLab result slide into comparable model-specific slides."""
from pathlib import Path
import win32com.client as win32

BASE=Path(__file__).resolve().parents[2]/'output'/'presentations'
DECK=BASE/'CrackLab_Five_Slide_Research_Presentation.pptx'
BACKUP=BASE/'CrackLab_Before_Expanded_Results.pptx'
def rgb(r,g,b):return r+(g<<8)+(b<<16)
INK=rgb(25,27,32);MUTED=rgb(87,94,106);PANEL=rgb(240,242,245)
BAR=rgb(190,196,205);ACCENT=rgb(84,93,112);LINE=rgb(207,211,218)

app=win32.GetActiveObject('PowerPoint.Application')
deck=next((app.Presentations(i) for i in range(1,app.Presentations.Count+1)
           if app.Presentations(i).Name.lower()==DECK.name.lower()),None)
if deck is None:raise RuntimeError('Open the CrackLab PPTX in PowerPoint')
if deck.Slides.Count!=6:raise RuntimeError('Expected the current six-slide deck')
deck.SaveCopyAs(str(BACKUP))
original=deck.Slides(5)
for _ in range(4):original.Duplicate()
if deck.Slides.Count!=10:raise RuntimeError('Slide duplication failed')

def rect(slide,name,x,y,w,h,fill,stroke=None):
    a=slide.Shapes.AddShape(1,x,y,w,h);a.Name=name;a.Fill.Solid();a.Fill.ForeColor.RGB=fill
    a.Line.Visible=1 if stroke is not None else 0
    if stroke is not None:a.Line.ForeColor.RGB=stroke;a.Line.Weight=.7
    return a
def text(slide,name,value,x,y,w,h,size=13,bold=False,color=INK,align=1):
    a=slide.Shapes.AddTextbox(1,x,y,w,h);a.Name=name;t=a.TextFrame
    t.MarginLeft=t.MarginRight=1;t.MarginTop=t.MarginBottom=0;t.WordWrap=True
    t.TextRange.Text=value;f=t.TextRange.Font;f.Name='Arial';f.Size=size;f.Bold=bold;f.Color.RGB=color
    t.TextRange.ParagraphFormat.Alignment=align
    return a
def reset(slide,title,number):
    for i in range(slide.Shapes.Count,6,-1):slide.Shapes(i).Delete()
    slide.Shapes(4).TextFrame.TextRange.Text=title
    slide.Shapes(6).TextFrame.TextRange.Text=f'CrackLab  /  {number:02d}'

models=[
 ('U-Net',24.60,15.75,81.32,'1,965,569',
  'High source Dice, but the weakest target Dice in this comparison. The change in dataset exposes a substantial domain gap.'),
 ('Paper-inspired R2AU',34.52,22.92,81.42,'2,110,029',
  'The recurrent and attention reference gains 9.92 target Dice points over U-Net, with roughly 2.11 million parameters.'),
 ('LiteCrack',34.01,22.65,76.36,'339,265',
  'Target Dice is 0.51 point below R2AU with 83.92% fewer parameters; source Dice is 5.06 points lower.'),
 ('LiteCrack-BA',41.72,28.91,75.40,'339,282',
  'Boundary supervision adds 7.71 target Dice points over LiteCrack. Source Dice decreases by about 0.96 point.'),
]
names=['U-Net','R2AU','LiteCrack','LiteCrack-BA']
values=[24.60,34.52,34.01,41.72]
for index,(name,dice,iou,source,parameters,interpretation) in enumerate(models):
    sl=deck.Slides(5+index)
    reset(sl,f'{name} Results',4+index)
    text(sl,'Protocol','DeepCrack source holdout → CrackMap target query  |  static checkpoints',80,131,800,24,14,False,MUTED)
    text(sl,'Target Dice label','TARGET DICE',82,178,305,21,13,True,MUTED)
    text(sl,'Target Dice value',f'{dice:.2f}%',80,204,350,73,49,True,ACCENT)
    text(sl,'Target IoU label','Target IoU',83,295,130,20,13,False,MUTED)
    text(sl,'Target IoU value',f'{iou:.2f}%',83,320,135,39,25,True,INK)
    text(sl,'Source Dice label','Source Dice',260,295,130,20,13,False,MUTED)
    text(sl,'Source Dice value',f'{source:.2f}%',260,320,135,39,25,True,INK)
    text(sl,'Parameters',f'{parameters} parameters',82,367,315,20,13,False,MUTED)
    rect(sl,'Divider',425,180,1,208,LINE)
    text(sl,'Chart title','TARGET QUERY DICE (%)',458,177,380,22,14,True,MUTED)
    for j,(label,value) in enumerate(zip(names,values)):
        y=214+j*40
        text(sl,f'Model {j}',label,458,y,110,22,12,j==index,INK)
        rect(sl,f'Bar track {j}',568,y+3,225,16,rgb(235,237,240))
        rect(sl,f'Dice bar {j}',568,y+3,225*value/45,16,ACCENT if j==index else BAR)
        text(sl,f'Dice label {j}',f'{value:.2f}',799,y,54,22,12,j==index,INK)
    rect(sl,'Takeaway background',79,401,800,55,PANEL)
    text(sl,'Takeaway',interpretation,92,411,773,38,16,False,INK)
    text(sl,'Evaluation note','Same 237 source and 95 target pairs; threshold 0.50. Single seed; target query inspected in earlier phases.',81,460,795,16,10,False,MUTED)
    sl.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
        f'{name} static result. Source Dice {source:.2f}%, target Dice {dice:.2f}%, '
        f'target IoU {iou:.2f}%, parameters {parameters}. Same 237-image DeepCrack '
        'source holdout and 95-image CrackMap target query. Per-image macro means at '
        'threshold 0.50; single training seed. The target query was inspected in '
        'earlier project phases. Sources: docs/FINAL_REPORT.md, sections Static '
        'comparison and System design; MODEL_CARD.md.'
    )

sl=deck.Slides(9)
reset(sl,'Six-Source Test Results',8)
text(sl,'Separate experiment','Separate LiteCrack model  |  512 px  |  496 publisher source-holdout images',80,130,800,26,14,False,MUTED)
text(sl,'Overall Dice label','OVERALL MEAN DICE',82,182,330,22,13,True,MUTED)
text(sl,'Overall Dice','28.73%',80,210,360,73,48,True,ACCENT)
text(sl,'Overall IoU','IoU 19.72%  ·  recall 27.02%',82,292,350,25,17,False,INK)
text(sl,'Different test note','Different dataset and protocol from the four-model comparison',82,341,340,42,15,False,MUTED)
rect(sl,'Divider',425,180,1,205,LINE)
text(sl,'Domain title','DICE BY CAPTURE SOURCE (%)',458,177,398,22,14,True,MUTED)
sources=[('GoPro',69.36),('iPhone',71.98),('UAV 3–10 m',21.67),
         ('UAV 15–30 m',20.85),('UAV 35–100 m',14.24),('Other UAV',14.59)]
for j,(label,value) in enumerate(sources):
    y=210+j*29
    text(sl,f'Source {j}',label,458,y,119,19,10,False,INK)
    rect(sl,f'Source track {j}',579,y+3,201,12,rgb(235,237,240))
    rect(sl,f'Source bar {j}',579,y+3,201*value/75,12,ACCENT if j<2 else BAR)
    text(sl,f'Score {j}',f'{value:.2f}',786,y,61,18,10,False,INK)
rect(sl,'Caveat background',79,401,800,55,PANEL)
text(sl,'Caveat','Ground-camera scores are much stronger than UAV scores; distant cracks remain a major limitation.',92,411,773,38,16,False,INK)
text(sl,'Test note','Validation-selected checkpoint was frozen before the official test was opened.',81,460,795,17,10,False,MUTED)
sl.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
    'Separate six-archive LiteCrack model, not directly comparable with the '
    'DeepCrack-to-CrackMap models. Official publisher source-holdout test: '
    '496 images; mean per-image Dice 0.287325, IoU 0.197227, recall 0.270186. '
    'Source Dice: GoPro 0.693625, iPhone 0.719845, UAV 3–10m 0.216736, '
    'UAV 15–30m 0.208454, UAV 35–100m 0.142357, other UAV 0.145943. '
    'Sources: output/multisource_test_v1/FINAL_TEST_REPORT.md and FROZEN_SELECTION.json.'
)

# Keep the pre-existing Future Scope slide after all result slides.
future=deck.Slides(10)
future.Shapes(6).TextFrame.TextRange.Text='CrackLab  /  09'
deck.Save()
print('Saved',DECK,'with',deck.Slides.Count,'slides')
