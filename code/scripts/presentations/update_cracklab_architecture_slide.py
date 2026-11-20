"""Replace the Proposed System slide's text boxes with editable architecture shapes."""
from pathlib import Path
import win32com.client as win32

BASE = Path(__file__).resolve().parents[2] / 'output' / 'presentations'
DECK = BASE / 'CrackLab_Five_Slide_Research_Presentation.pptx'
BACKUP = BASE / 'CrackLab_Five_Slide_Research_Presentation_before_architecture.pptx'
PDF = BASE / 'CrackLab_Five_Slide_Research_Presentation.pdf'

def rgb(r,g,b): return r + (g << 8) + (b << 16)
INK=rgb(28,32,39); MUTED=rgb(77,85,98); GREEN=rgb(222,238,221)
BLUE=rgb(218,231,242); PURPLE=rgb(231,220,241); GRAY=rgb(239,241,244)
TEAL=rgb(53,135,137); LINE=rgb(97,107,120)

app=win32.GetActiveObject('PowerPoint.Application')
deck=None
for i in range(1,app.Presentations.Count+1):
    item=app.Presentations(i)
    if item.Name.lower()==DECK.name.lower(): deck=item;break
if deck is None: raise RuntimeError('The CrackLab presentation is not open in PowerPoint')
if deck.Slides.Count!=6: raise RuntimeError('Expected six slides, including the title slide')
if not BACKUP.exists(): deck.SaveCopyAs(str(BACKUP))
slide=deck.Slides(3)
for i in range(slide.Shapes.Count,6,-1): slide.Shapes(i).Delete()

def shape(name,x,y,w,h,fill,stroke=None,radius=True):
    s=slide.Shapes.AddShape(5 if radius else 1,x,y,w,h)
    s.Name=name;s.Fill.Solid();s.Fill.ForeColor.RGB=fill
    s.Line.Visible=1 if stroke is not None else 0
    if stroke is not None:s.Line.ForeColor.RGB=stroke;s.Line.Weight=.9
    return s

def label(name,value,x,y,w,h,size=13,bold=False,color=INK,align=2):
    s=slide.Shapes.AddTextbox(1,x,y,w,h);s.Name=name
    t=s.TextFrame;t.MarginLeft=t.MarginRight=2;t.MarginTop=t.MarginBottom=0
    t.WordWrap=True;t.TextRange.Text=value
    f=t.TextRange.Font;f.Name='Arial';f.Size=size;f.Bold=bold;f.Color.RGB=color
    t.TextRange.ParagraphFormat.Alignment=align
    return s

def line(name,x1,y1,x2,y2,color=LINE,width=1.5,arrow=False):
    c=slide.Shapes.AddConnector(1,x1,y1,x2,y2);c.Name=name
    c.Line.ForeColor.RGB=color;c.Line.Weight=width
    if arrow:c.Line.EndArrowheadStyle=3
    return c

def block(code,desc,x,y,w,fill):
    shape(code,x,y,w,45,fill,LINE)
    label(code+' label',code,x+2,y+4,w-4,18,12,True)
    label(code+' detail',desc,x+2,y+23,w-4,16,9,False,MUTED)

label('Architecture subtitle','LiteCrack / LiteCrack-BA: U-shaped segmentation network',80,128,540,28,16,True,INK,1)
label('Input heading','RGB image',83,185,74,18,11,True)
shape('RGB input',89,212,62,45,GRAY,LINE)
label('RGB input label','256 × 256',91,224,58,16,10,False,MUTED)

enc=[('E1','feature 1',179,211),('E2','feature 2',259,242),('E3','feature 3',339,273),('E4','feature 4',419,304),('E5','deep feature',499,335)]
for code,desc,x,y in enc:block(code,desc,x,y,66,GREEN)
label('Encoder title','MobileNetV3-Small ½-width encoder  ·  GroupNorm',179,185,390,19,11,True,MUTED,1)

dec=[('D1','128 ch',579,304),('D2','64 ch',659,273),('D3','32 ch',739,242),('D4','16 ch',819,211)]
for code,desc,x,y in dec:block(code,desc,x,y,66,BLUE)
label('Decoder title','Nearest upsample + skip concat + separable conv',579,185,306,19,11,True,MUTED,1)

path=[(151,234,179,234),(245,234,259,264),(325,264,339,295),(405,295,419,326),
      (485,326,499,357),(565,357,579,326),(645,326,659,295),(725,295,739,264),(805,264,819,234)]
for i,coords in enumerate(path):line('forward %02d'%i,*coords,TEAL,2.1,True)

# Four corresponding encoder-to-decoder skip paths. Each joins a decoder input.
for i,(sx,sy,dx) in enumerate([(245,221,819),(325,252,739),(405,283,659),(485,314,579)]):
    line(f'skip {i+1}',sx,sy,dx,sy,rgb(128,105,161),1.25,True)

label('Skip legend','purple = encoder skip features',81,391,235,20,10,False,rgb(128,105,161),1)
label('Decoder legend','D1–D4 are four decoder blocks',344,391,245,20,10,False,MUTED,1)

# Shared refinement features feed the region head and the optional BA head.
line('decoder to refine',852,211,852,151,TEAL,1.8,True)
shape('Refine features',814,123,75,28,PURPLE,LINE)
label('Refine label','Refine',816,129,71,16,11,True)
line('head branch rise',852,123,852,112,TEAL,1.4)
line('head branch span',852,112,684,112,TEAL,1.4)
line('region head',684,112,684,121,TEAL,1.4,True)
line('boundary head',764,112,764,121,TEAL,1.4,True)
shape('Region mask',647,121,77,32,GRAY,LINE)
label('Region mask text','Crack mask',649,128,73,17,10,True)
shape('Boundary mask',727,121,74,32,PURPLE,LINE)
label('Boundary mask text','Boundary*',729,128,70,17,10,True)
label('BA footnote','* LiteCrack-BA only',647,156,150,13,9,False,MUTED,1)

# Application workflow is downstream of model prediction, not part of the network.
shape('Review workflow background',80,422,799,38,GRAY,None)
label('Review workflow','CrackLab application: inspect prediction  →  brush correction  →  compare / export',93,430,772,23,14,True,INK,1)
slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
    'Architecture follows src/litecrack.py: MobileNetV3-Small width_mult=.5; feature indices '
    '0, 1, 3, 8, 12; four nearest-neighbor upsample + concatenation + SeparableBlock '
    'decoder stages with widths 128, 64, 32, 16; refine block; 1x1 region head and '
    'optional 1x1 boundary head. Diagram is schematic; boxes are not drawn to tensor scale. '
    'CrackLab review workflow is downstream of inference. Source: src/litecrack.py; MODEL_CARD.md.'
)
deck.Save()
print('Updated architecture slide in',DECK)
