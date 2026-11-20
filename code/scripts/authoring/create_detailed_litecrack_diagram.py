"""Draw the shared LiteCrack network and its two training variants."""
from pathlib import Path
from xml.sax.saxutils import escape

out = Path("output/diagrams/litecrack_multisource_detailed.svg")
out.parent.mkdir(parents=True, exist_ok=True)
p = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1050" viewBox="0 0 1800 1050">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#637b85"/></marker><marker id="skip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#13967d"/></marker></defs>
<rect width="1800" height="1050" fill="#f7faf9"/>
<style>text{font-family:Arial,sans-serif;fill:#193038}.title{font-size:39px;font-weight:700}.sub{font-size:19px;fill:#53717a}.head{font-size:18px;font-weight:700}.mid{font-size:16px}.note{font-size:14px;fill:#5b747c}.arr{fill:none;stroke:#637b85;stroke-width:2.7;marker-end:url(#arrow)}.skip{fill:none;stroke:#13967d;stroke-width:2.5;stroke-dasharray:7 6;marker-end:url(#skip)}</style>''']
def text(x,y,s,cls="mid",anchor="start"):
    p.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')
def rect(x,y,w,h,fill="#fff",stroke="#c9d8db"):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def line(d,cls="arr"):
    p.append(f'<path d="{d}" class="{cls}"/>')
def box(x,y,w,h,title,detail,fill,stroke):
    rect(x,y,w,h,fill,stroke)
    text(x+w/2,y+30,title,"head","middle")
    text(x+w/2,y+53,detail,"note","middle")

text(55,67,"LiteCrack: model architecture","title")
text(55,100,"Original and multi-source checkpoints use the same network graph","sub")
rect(1320,32,425,78,"#e7f4ef","#b7dbce")
text(1340,65,"Shared architecture","head")
text(1340,91,"Separate training data and learned weights","note")

# Image notation and encoder feature maps.
rect(45,186,158,178,"#e6ecec","#9eb5b9")
p.append('<path d="M59 338L92 275L122 305L155 237L189 338Z" fill="#8da6a7"/><path d="M60 335Q125 295 188 335" fill="none" stroke="#263b40" stroke-width="9"/>')
text(124,218,"RGB input","head","middle")
text(124,390,"256 × 256 × 3","note","middle")
line("M204 272H240")
enc=[(250,190,170,166,"E0 · index 0","early feature map"),(455,210,160,146,"E1 · index 1","mobile block"),(650,230,150,126,"E2 · index 3","mobile blocks"),(835,250,140,106,"E3 · index 8","mobile blocks"),(1010,270,130,86,"E4 · index 12","deep feature map")]
text(704,161,"½-width MobileNetV3-Small encoder · GroupNorm · no pretrained weights","head","middle")
for i,(x,y,w,h,title,detail) in enumerate(enc):
    box(x,y,w,h,title,detail,"#e0edf7","#8bb6d8")
    if i<4: line(f"M{x+w} 273H{enc[i+1][0]-8}")

# Four decoder stages, deep to shallow. Exact widths and skip indices come from src/litecrack.py.
dec=[(1020,478,155,112,"D1 · 128 ch","skip from E3"),(825,503,155,112,"D2 · 64 ch","skip from E2"),(630,528,155,112,"D3 · 32 ch","skip from E1"),(435,553,155,112,"D4 · 16 ch","skip from E0")]
line("M1075 356V464")
for i,(x,y,w,h,title,detail) in enumerate(dec):
    box(x,y,w,h,title,detail,"#e0f4ec","#78cbb6")
    if i<3: line(f"M{x} {y+55}H{dec[i+1][0]+dec[i+1][2]+8}")
for (ex,ey,ew,eh,*_), (dx,dy,dw,dh,*_) in zip(enc[3::-1],dec):
    line(f"M{ex+ew/2:.0f} {ey+eh}V{dy-23}H{dx+dw/2:.0f}V{dy-8}","skip")
text(815,699,"Nearest resize → concatenate matching encoder skip → depthwise-separable block","mid","middle")

line("M435 610H395")
box(220,553,165,112,"Refine · 16 ch","full input size","#fff0df","#e8bb85")
text(302,540,"nearest resize","note","middle")
line("M220 610H205")
box(35,553,160,112,"1 × 1 conv","one logit / pixel","#fff0df","#e8bb85")
line("M115 665V708")
box(35,715,235,85,"Sigmoid → mask","256 × 256 × 1","#e5f0e9","#a5c8b3")

# Inner operations for one of the decoder/refine blocks.
rect(310,733,1435,224)
text(337,773,"Inside one depthwise-separable block","head")
ops=[("DW 3 × 3","per channel"),("PW 1 × 1","channel mix"),("GroupNorm","one group"),("ReLU","activation"),("DW 3 × 3","per channel"),("PW 1 × 1","channel mix"),("GroupNorm","one group"),("ReLU","activation")]
for i,(title,detail) in enumerate(ops):
    x=337+i*175
    box(x,807,155,87,title,detail,"#f0f6f5","#bdd0cf")
    if i<7: line(f"M{x+157} 850H{x+171}")
text(337,930,"The decoder concatenates the resized decoder feature and encoder skip before this block.","note")

rect(35,976,830,60,"#fff7f0","#efc8aa")
rect(890,976,855,60,"#edf8f3","#b5daca")
text(55,1003,"Original LiteCrack","head")
text(55,1026,"Source road-crack training checkpoint","note")
text(910,1003,"Multi-source LiteCrack","head")
text(910,1026,"Same graph; six-source training, 2,526 train / 542 validation images","note")
p.append("</svg>")
out.write_text("\n".join(p),encoding="utf-8")
print(out.resolve())
