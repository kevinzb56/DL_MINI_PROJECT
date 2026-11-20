"""Reference-style, editable two-panel schematic for CrackLab slide 3."""
from pathlib import Path
from PIL import Image
import win32com.client as win32

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output'/'presentations'
DECK=BASE/'CrackLab_Five_Slide_Research_Presentation.pptx'
FIG=ROOT/'output'/'phase8'/'figures'/'real_demo.png'
BACKUP=BASE/'CrackLab_Five_Slide_Research_Presentation_before_reference_style.pptx'
PHOTO=BASE/'cracklab_architecture_input.png'
MASK=BASE/'cracklab_architecture_reference_mask.png'
PRED=BASE/'cracklab_architecture_prediction_overlay.png'
im=Image.open(FIG)
im.crop((8,88,532,612)).save(PHOTO)
im.crop((1088,88,1612,612)).save(MASK)
im.crop((548,88,1072,612)).save(PRED)

def rgb(r,g,b):return r+(g<<8)+(b<<16)
BLACK=rgb(30,32,37);GRAY=rgb(87,92,103);BORDER=rgb(92,99,108)
GREEN=rgb(206,232,191);YELLOW=rgb(238,188,41);ORANGE=rgb(211,103,37)
CYAN=rgb(43,175,183);PURPLE=rgb(144,107,187);BLUE=rgb(34,89,161)
RED=rgb(208,46,45);DARK=rgb(54,61,77)

app=win32.GetActiveObject('PowerPoint.Application')
deck=next((app.Presentations(i) for i in range(1,app.Presentations.Count+1)
           if app.Presentations(i).Name.lower()==DECK.name.lower()),None)
if deck is None:raise RuntimeError('Open the CrackLab PPTX in PowerPoint')
if deck.Slides.Count!=6:raise RuntimeError('Expected six slides')
if not BACKUP.exists():deck.SaveCopyAs(str(BACKUP))
s=deck.Slides(3)
for i in range(s.Shapes.Count,6,-1):s.Shapes(i).Delete()

def rect(name,x,y,w,h,fill=rgb(255,255,255),stroke=BORDER,rounded=False):
    a=s.Shapes.AddShape(5 if rounded else 1,x,y,w,h);a.Name=name
    a.Fill.Solid();a.Fill.ForeColor.RGB=fill
    a.Line.Visible=1 if stroke is not None else 0
    if stroke is not None:a.Line.ForeColor.RGB=stroke;a.Line.Weight=.75
    return a
def text(name,value,x,y,w,h,size=10,bold=False,color=BLACK,align=2):
    a=s.Shapes.AddTextbox(1,x,y,w,h);a.Name=name;t=a.TextFrame
    t.MarginLeft=t.MarginRight=1;t.MarginTop=t.MarginBottom=0;t.WordWrap=True
    t.TextRange.Text=value;f=t.TextRange.Font;f.Name='Arial';f.Size=size;f.Bold=bold;f.Color.RGB=color
    t.TextRange.ParagraphFormat.Alignment=align
    return a
def edge(name,x1,y1,x2,y2,color=BORDER,width=1.4,arrow=True):
    a=s.Shapes.AddConnector(1,x1,y1,x2,y2);a.Name=name
    a.Line.ForeColor.RGB=color;a.Line.Weight=width
    if arrow:a.Line.EndArrowheadStyle=3
    return a
def dot(name,x,y,r,fill,stroke=BORDER):
    return rect(name,x-r,y-r,2*r,2*r,fill,stroke,False) if False else circle(name,x,y,r,fill,stroke)
def circle(name,x,y,r,fill,stroke=BORDER):
    a=s.Shapes.AddShape(9,x-r,y-r,2*r,2*r);a.Name=name
    a.Fill.Solid();a.Fill.ForeColor.RGB=fill
    a.Line.Visible=1 if stroke is not None else 0
    if stroke is not None:a.Line.ForeColor.RGB=stroke;a.Line.Weight=.75
    return a
def picture(name,path,x,y,w,h):
    a=s.Shapes.AddPicture(str(path),False,True,x,y,w,h);a.Name=name
    a.Line.Visible=1;a.Line.ForeColor.RGB=BORDER;a.Line.Weight=.75
    return a

# The reference uses two labeled schematics and a compact symbol key.
text('Panel A title','(a) LiteCrack-BA segmentation architecture',79,124,543,21,13,True,BLACK,1)
text('Panel B title','(b) Few-shot refinement and review',640,124,250,21,13,True,BLACK,1)
edge('Panel divider',630,146,630,456,rgb(190,193,198),.8,False)

# Actual project photograph and reference-mask crop establish the input/output notation.
picture('Input RGB image',PHOTO,80,170,46,46)
text('Input label','RGB image',78,220,53,20,8,False,BLACK)
text('Output label','mask example',570,117,53,13,7,False,BLACK)
picture('Output mask',MASK,575,140,43,43)

# Five feature levels from encoder indices 0,1,3,8,12. No attention or recurrent modules.
nodes=[('8 ch','128²',145,185),('8 ch','64²',215,220),('16 ch','32²',285,255),
       ('24 ch','16²',355,290),('288 ch','8²',425,325),('128 ch','16²',480,290),
       ('64 ch','32²',535,255),('32 ch','64²',565,220),('16 ch','128²',565,185)]
for j,(channels,spatial,x,y) in enumerate(nodes):
    # Three offset feature planes and a narrow convolution bar echo the source figure.
    fill=rgb(209,233,196) if j<5 else rgb(215,235,207)
    for plane in range(3):
        rect(f'feature {j+1} plane {plane+1}',x+plane*5,y+plane*2,23,22,fill,BORDER)
    rect(f'feature {j+1} conv',x+27,y+3,7,22,rgb(157,208,139),BORDER)
    if j==8:
        text(f'feature {j+1} channels',channels.replace(' ','') ,x+37,y+4,26,13,7,False,BLACK,1)
    else:
        text(f'feature {j+1} channels',channels,x-5,y-15,44,13,7,False,BLACK)
    text(f'feature {j+1} size',spatial,x-5,y+27,44,12,7,False,GRAY)
for j in range(4):
    ex,ey=nodes[j][2:];nx,ny=nodes[j+1][2:]
    edge(f'downsample {j+1}',ex+36,ey+13,nx,ny+13,YELLOW,2.1,True)
edge('encoder bottleneck',461,338,480,303,YELLOW,2.1,True)
for j in range(5,8):
    ex,ey=nodes[j][2:];nx,ny=nodes[j+1][2:]
    edge(f'upsample {j-4}',ex+36,ey+13,nx,ny+13,ORANGE,2,True)
edge('input to encoder',126,193,145,198,rgb(81,151,61),1.8,True)
edge('decoder to output',601,198,608,193,rgb(81,151,61),1.8,True)

# Cyan skip paths terminate in purple concatenate symbols at decoder stages.
for k,(x1,y1,x2,y2) in enumerate([(181,198,565,198),(251,233,565,233),
                                    (321,268,535,268),(391,303,480,303)]):
    edge(f'skip feature {k+1}',x1,y1,x2-7,y2,CYAN,1.8,True)
    circle(f'concat symbol {k+1}',x2-5,y2,5,rgb(229,216,242),PURPLE)
    text(f'concat plus {k+1}','+',x2-9,y2-6,8,9,7,True,PURPLE)

# Shared refinement creates region logits and the optional auxiliary boundary logits.
rect('Refine block',502,149,48,22,rgb(229,216,242),PURPLE,True)
text('Refine text','refine',504,154,44,13,8,True)
edge('D4 depart',565,185,558,185,PURPLE,1.3,False)
edge('D4 rise',558,185,558,135,PURPLE,1.3,False)
edge('D4 span',558,135,526,135,PURPLE,1.3,False)
edge('D4 to refine',526,135,526,149,PURPLE,1.3,True)
edge('refine to region',550,160,575,160,PURPLE,1.2,True)
rect('Auxiliary head',502,179,48,18,rgb(229,216,242),PURPLE,True)
text('Auxiliary head label','BA edge*',504,181,44,13,7,True,PURPLE)
edge('boundary branch',526,171,526,179,PURPLE,1.2,True)

# A small module inset, like the reference's internal block diagram.
rect('Block inset',80,282,123,91,rgb(243,246,240),BORDER)
text('Block heading','Separable decoder block',85,286,112,15,9,True)
rect('Depthwise conv',89,309,49,22,GREEN,BORDER)
rect('Pointwise conv',145,309,49,22,GREEN,BORDER)
text('Depthwise label','3×3 DW',90,315,47,13,7,True)
text('Pointwise label','1×1 PW',146,315,47,13,7,True)
edge('module internal',138,320,145,320,rgb(81,151,61),1.4,True)
text('module lower','GroupNorm + ReLU, repeated',85,343,112,23,8,False,GRAY)

# Reference-style legend with actual semantics.
legend=[(211,392,GREEN,'feature maps'),(310,392,YELLOW,'downsample'),
        (419,392,ORANGE,'upsample'),(520,392,CYAN,'skip')]
for j,(x,y,col,caption) in enumerate(legend):
    rect(f'legend swatch {j}',x,y,17,9,col,None)
    text(f'legend label {j}',caption,x+20,y-2,92,14,8,False,BLACK,1)
circle('legend concat',218,419,5,rgb(229,216,242),PURPLE)
text('legend concat label','concat',230,414,55,14,8,False,BLACK,1)
text('legend note','Nearest upsampling; no attention gates in LiteCrack',309,414,312,18,8,False,GRAY,1)

# Right panel follows the reference's training -> prediction -> inspection -> refinement loop.
rect('static learning header',649,153,107,19,rgb(245,185,64),None,True)
text('static learning label','Static training',651,156,103,13,8,True)
for j in range(3):picture(f'training sample {j}',MASK,658+j*20,180,33,33)
text('training samples label','DeepCrack train masks',649,217,110,17,8,False,GRAY)
edge('training to model',754,196,775,196,ORANGE,2,True)
for j,(x,y) in enumerate([(780,185),(798,176),(798,196),(815,185)]):circle(f'model neuron {j}',x,y,6,rgb(235,242,252),BLUE)
for j,(a,b,c,d) in enumerate([(786,185,792,176),(786,185,792,196),(804,176,809,185),(804,196,809,185)]):
    edge(f'neuron connection {j}',a,b,c,d,BLUE,1,False)
text('model name','LiteCrack-BA',772,213,85,15,8,True,BLUE)
edge('model predicts',820,196,841,196,BLUE,2,True)
picture('prediction overlay',PRED,843,176,36,36)
text('prediction label','prediction',838,216,52,15,8,False,GRAY)

circle('review accepted',785,263,11,rgb(90,160,75),None)
text('accepted symbol','✓',779,255,13,16,11,True,rgb(255,255,255))
text('accepted label','accept',765,280,42,18,8,False,BLACK)
circle('review correction',854,263,11,RED,None)
text('correction symbol','×',848,255,13,16,12,True,rgb(255,255,255))
text('correction label','correct',832,280,44,18,8,False,BLACK)
edge('prediction decision stem',861,213,861,239,BLUE,1.4,False)
edge('accepted branch',861,239,785,252,rgb(90,160,75),1.3,True)
edge('correction branch',861,239,854,252,RED,1.3,True)

rect('review editor',659,332,222,42,rgb(241,244,247),BORDER)
text('review editor text','CrackLab: brush correction + compare',666,342,207,22,9,True,BLACK,1)
edge('correction to editor',854,274,854,332,RED,1.4,True)
rect('few-shot refinement',660,398,221,36,rgb(222,239,232),BORDER)
text('few-shot refinement text','K = 1 / 5 / 10 support; decoder + heads update',666,406,210,22,8,True,BLACK,1)
edge('editor to fewshot',770,374,770,398,CYAN,2,True)
text('freeze note','encoder frozen · source retention checked',660,437,222,17,8,False,GRAY,1)

text('architecture caption','Model schematic (left) and reviewed few-shot adaptation workflow (right)',80,466,793,17,10,True,BLACK,1)
s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text=(
    'Reference-inspired two-panel figure. This is LiteCrack-BA, not the cited R2AU-Net. '
    'Left: MobileNetV3-Small width 0.5 encoder, five saved feature levels, four decoder '
    'blocks with nearest interpolation, skip concatenation and separable convolutions, '
    'shared refine features, region output and optional boundary head. Diagram is schematic '
    'but feature size/channel labels were checked with a 256x256 forward pass. Right: static training, prediction, '
    'human mask review/correction, and bounded few-shot adaptation that freezes encoder '
    'and updates decoder and heads with K=1/5/10 support. Sources: src/litecrack.py; '
    'docs/FINAL_REPORT.md; MODEL_CARD.md. Input/mask crops are from '
    'output/phase8/figures/real_demo.png, an existing dataset acquisition example. '
    'The left black/white mask is an annotation illustration; the right prediction '
    'thumbnail is the actual model overlay from that figure.'
)
deck.Save()
print('Updated',DECK)
