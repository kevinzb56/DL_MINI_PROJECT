"""Create the editable LiteCrack / multi-source architecture figure."""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path('output/diagrams/litecrack_multisource_architecture.svg')
OUT.parent.mkdir(parents=True, exist_ok=True)
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="920" viewBox="0 0 1600 920">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#83aaba"/></marker></defs>
<rect width="1600" height="920" fill="#0d1519"/>
<style>text{font-family:Arial,sans-serif;fill:#e8f1f1}.title{font-size:39px;font-weight:700}.sub{font-size:19px;fill:#a6bdc4}.label{font-size:19px;font-weight:700}.small{font-size:16px;fill:#b5cbd0}.tiny{font-size:14px;fill:#88a5ae}.flow{fill:none;stroke:#83aaba;stroke-width:3;marker-end:url(#arrow)}.skip{fill:none;stroke:#56bda3;stroke-width:2;stroke-dasharray:7 6;marker-end:url(#arrow)}</style>''']

def text(x,y,s,klass='small',anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" class="{klass}" text-anchor="{anchor}">{escape(s)}</text>')
def box(x,y,w,h,title,lines=(),fill='#1a282e',stroke='#34505a',accent=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    if accent: parts.append(f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3" fill="{accent}"/>')
    text(x+18,y+31,title,'label')
    for i,line in enumerate(lines): text(x+18,y+57+i*22,line,'small')
def arrow(x1,y1,x2,y2,klass='flow'):
    parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" class="{klass}"/>')

text(64,76,'LiteCrack  /  Multi-source LiteCrack','title')
text(64,111,'Shared segmentation network • different training data and checkpoints','sub')

box(64,157,710,145,'01  LiteCrack',('Original compact model','Source road-crack images and masks → train checkpoint'),accent='#ff9563')
box(826,157,710,145,'02  Multi-source LiteCrack',('Same model, trained across six capture sources','2,526 train images • 542 validation images • seed 42'),accent='#56bda3')
arrow(419,302,419,350); arrow(1181,302,1181,350)

box(64,350,710,83,'Source image → RGB input',('Resize to 256 × 256 for the original training protocol',),accent='#ff9563')
box(826,350,710,83,'Six-source image → RGB input',('Multi-source training config: 256 × 256',),accent='#56bda3')
arrow(419,433,419,485); arrow(1181,433,1181,485)

parts.append('<rect x="64" y="485" width="1472" height="275" rx="18" fill="#152329" stroke="#3a5963" stroke-width="2"/>')
text(92,525,'SHARED NETWORK ARCHITECTURE','label')
text(92,550,'Half-width MobileNetV3-Small encoder • random initialization • GroupNorm','small')
box(92,586,290,104,'Encoder features',('Stages 0, 1, 3, 8, 12','Five saved feature maps'),fill='#20343b',accent='#73a8d7')
box(472,586,310,104,'Four decoder blocks',('Nearest upsample + skip concat','Depthwise-separable convolutions'),fill='#20343b',accent='#56bda3')
box(872,586,253,104,'Full-size refine',('Nearest upsample','Separable block'),fill='#20343b',accent='#56bda3')
box(1215,586,290,104,'Region prediction',('1 × 1 convolution → logits','Sigmoid → crack mask'),fill='#20343b',accent='#ff9563')
arrow(382,638,472,638); arrow(782,638,872,638); arrow(1125,638,1215,638)
text(795,723,'Four encoder skip features feed the decoder at matching scales','tiny','middle')

box(64,800,710,79,'LiteCrack output',('Pixel-level crack probability / binary mask',),accent='#ff9563')
box(826,800,710,79,'Multi-source output',('Same prediction type, different learned weights',),accent='#56bda3')
arrow(419,760,419,800); arrow(1181,760,1181,800)
text(64,902,'Source: src/litecrack.py and output/multisource_full_v1/package_metadata/config.json','tiny')
parts.append('</svg>')
OUT.write_text('\n'.join(parts), encoding='utf-8')
print(OUT.resolve())
