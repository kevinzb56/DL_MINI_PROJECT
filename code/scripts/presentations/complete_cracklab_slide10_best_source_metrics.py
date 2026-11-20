"""Complete best per-source Dice, IoU, and recall on current slide 10."""
from pathlib import Path
import win32com.client as win32

BASE=Path(__file__).resolve().parents[2]/'output'/'presentations'
DECK=BASE/'CrackLab_Five_Slide_Research_Presentation.pptx'
BACKUP=BASE/'CrackLab_Before_Slide10_Best_Source_Metrics.pptx'
def rgb(r,g,b):return r+(g<<8)+(b<<16)
INK=rgb(25,27,32);MUTED=rgb(86,94,107);ACCENT=rgb(84,93,112)

app=win32.GetActiveObject('PowerPoint.Application')
deck=next((app.Presentations(i) for i in range(1,app.Presentations.Count+1)
           if app.Presentations(i).Name.lower()==DECK.name.lower()),None)
if deck is None:raise RuntimeError('Open the CrackLab deck in PowerPoint')
if deck.Slides.Count!=11:raise RuntimeError('Expected current eleven-slide deck')
s=deck.Slides(10)
if s.Shapes(4).TextFrame.TextRange.Text!='Six-Source Test Results':
    raise RuntimeError('Slide 10 is not Six-Source Test Results')
deck.SaveCopyAs(str(BACKUP))

def label(name,value,x,y,w,h,size,bold=False,color=INK):
    a=s.Shapes.AddTextbox(1,x,y,w,h);a.Name=name;t=a.TextFrame
    t.MarginLeft=t.MarginRight=1;t.MarginTop=t.MarginBottom=0;t.WordWrap=True
    t.TextRange.Text=value;f=t.TextRange.Font;f.Name='Arial';f.Size=size;f.Bold=bold;f.Color.RGB=color
    return a

# Keep the six-source bars and existing caveat; replace the left metric column.
for i in (11,10):s.Shapes(i).Delete()
heading=s.Shapes(8);heading.TextFrame.TextRange.Text='BEST MEAN DICE'
heading.Left=82;heading.Top=177;heading.Width=330;heading.Height=17
value=s.Shapes(9);value.TextFrame.TextRange.Text='71.98%'
value.Left=80;value.Top=196;value.Width=335;value.Height=49
value.TextFrame.TextRange.Font.Size=37
label('Best IoU label','BEST IoU',82,251,330,18,13,True,MUTED)
label('Best IoU value','56.52%',80,270,335,39,29,True,ACCENT)
label('Best Recall label','BEST RECALL',82,317,330,18,13,True,MUTED)
label('Best Recall value','73.20%',80,336,335,39,29,True,ACCENT)
label('Best source scope','Ground iPhone (50 images)  |  overall six-source Dice: 28.73%',82,381,340,20,11,False,MUTED)

s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
    'The three best archive-level macro means among the six publisher capture '
    'sources are all from ground iPhone (n=50): Dice 0.71984450340271, '
    'IoU 0.5652338629961013, recall 0.732012364268303. These are means of '
    'per-image scores at threshold 0.5 for the separate 512px LiteCrack model. '
    'Overall source-holdout (all 496 images) Dice is 0.2873248092622699, '
    'IoU 0.19722701263174625, recall 0.27018597552547186. Do not present '
    'the iPhone subset as the overall test result or compare it directly with '
    'the DeepCrack-to-CrackMap model results. Source: '
    'output/multisource_test_v1/summary.json and FINAL_TEST_REPORT.md.'
)
deck.Save()
print('Updated slide 10 in',DECK)
