"""Add scoped best static-model metrics to the final CrackLab slide."""
from pathlib import Path
import win32com.client as win32

BASE=Path(__file__).resolve().parents[2]/'output'/'presentations'
DECK=BASE/'CrackLab_Five_Slide_Research_Presentation.pptx'
BACKUP=BASE/'CrackLab_Before_Best_Metrics_Slide10.pptx'
def rgb(r,g,b):return r+(g<<8)+(b<<16)
INK=rgb(25,27,32);MUTED=rgb(86,94,107);ACCENT=rgb(84,93,112)
PANEL=rgb(240,242,245);LINE=rgb(207,211,218)

app=win32.GetActiveObject('PowerPoint.Application')
deck=next((app.Presentations(i) for i in range(1,app.Presentations.Count+1)
           if app.Presentations(i).Name.lower()==DECK.name.lower()),None)
if deck is None:raise RuntimeError('Open the CrackLab presentation in PowerPoint')
if deck.Slides.Count!=10:raise RuntimeError('Expected ten slides')
deck.SaveCopyAs(str(BACKUP))
s=deck.Slides(10)
for i in range(s.Shapes.Count,6,-1):s.Shapes(i).Delete()
s.Shapes(4).TextFrame.TextRange.Text='Best Results & Future Scope'

def rect(name,x,y,w,h,fill):
    a=s.Shapes.AddShape(1,x,y,w,h);a.Name=name;a.Fill.Solid();a.Fill.ForeColor.RGB=fill
    a.Line.Visible=0
    return a
def text(name,value,x,y,w,h,size,bold=False,color=INK,align=1):
    a=s.Shapes.AddTextbox(1,x,y,w,h);a.Name=name;t=a.TextFrame
    t.MarginLeft=t.MarginRight=1;t.MarginTop=t.MarginBottom=0;t.WordWrap=True
    t.TextRange.Text=value;f=t.TextRange.Font;f.Name='Arial';f.Size=size;f.Bold=bold;f.Color.RGB=color
    t.TextRange.ParagraphFormat.Alignment=align
    return a

text('Scope','LiteCrack-BA  |  best static model on the 95-image CrackMap target query',80,131,800,26,15,False,MUTED)
metrics=[('BEST MEAN DICE','41.72%'),('BEST IoU','28.91%'),('BEST RECALL','32.04%')]
for i,(label,value) in enumerate(metrics):
    x=80+i*268
    rect(f'Metric panel {i+1}',x,174,250,116,PANEL)
    text(f'Metric label {i+1}',label,x+15,189,220,20,13,True,MUTED)
    text(f'Metric value {i+1}',value,x+15,217,220,55,35,True,ACCENT)

text('Future scope heading','Next steps',80,314,800,27,19,True,INK)
rect('Future scope divider',80,346,800,1,LINE)
steps=[
    'Improve UAV crack detection with fully annotated images from varied heights.',
    'Validate on new physical sites and cameras without tuning on the frozen test.',
    'Measure human correction time and remaining mask errors in CrackLab.',
]
for i,step in enumerate(steps):
    y=358+i*32
    text(f'Step number {i+1}',f'{i+1:02d}',82,y,34,23,13,True,ACCENT)
    text(f'Step text {i+1}',step,123,y,743,26,14,False,INK)
text('Protocol note','Per-image macro means at threshold 0.50. Separate six-source and few-shot results are excluded.',81,464,795,17,10,False,MUTED)
s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
    'Best values are among the four static models evaluated on the same 95-image '
    'CrackMap target query, not across separate protocols. LiteCrack-BA: '
    'Dice 0.4171907729420223, IoU 0.28913149288610407, '
    'recall 0.3203956855951171. These are per-image macro means at '
    'threshold 0.50. Source: output/phase8/evidence.json static_models target; '
    'docs/FINAL_REPORT.md. The adapted K=10 boundary model has mean target Dice '
    '59.09% over five support draws, but no matching IoU/recall summary in this '
    'three-metric static comparison. The separate six-source test uses different data.'
)
deck.Save()
print('Updated slide 10 in',DECK)
