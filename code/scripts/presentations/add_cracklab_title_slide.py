from pathlib import Path
from shutil import copy2
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

base = Path(__file__).resolve().parents[2] / 'output' / 'presentations'
deck_path = base / 'CrackLab_Five_Slide_Research_Presentation.pptx'
backup_path = base / 'CrackLab_Five_Slide_Research_Presentation_before_title.pptx'
bg_path = base / 'template_background.png'
copy2(deck_path, backup_path)
prs = Presentation(deck_path)
assert len(prs.slides) == 5, 'Expected the original five-slide deck'
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.shapes.add_picture(str(bg_path), 0, 0, prs.slide_width, prs.slide_height)

def rect(x, y, w, h, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*color)
    shape.line.fill.background()

def text(value, x, y, w, h, size, bold=False, color=(25, 25, 25), font='Arial', align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(.02)
    frame.margin_top = frame.margin_bottom = Inches(.02)
    para = frame.paragraphs[0]
    para.text = value
    para.font.name = font
    para.font.size = Pt(size)
    para.font.bold = bold
    para.font.color.rgb = RGBColor(*color)
    if align is not None:
        para.alignment = align

rect(.55, .38, 12.22, 6.73, (247, 247, 247))
rect(.72, .55, 11.9, 6.36, (255, 255, 255))
text('Research Project', 1.1, 1.14, 11.1, .42, 19, False, (99, 104, 113))
text('CrackLab', 1.08, 1.83, 11.12, 1.02, 52, True, (15, 15, 15), 'Arial Narrow')
rect(1.1, 3.02, 11.0, .03, (95, 102, 115))
text('Road crack segmentation and human review', 1.1, 3.42, 10.7, 1.15, 29, False, (65, 70, 79))
text('DeepCrack → CrackMap research study', 1.1, 5.52, 10.7, .42, 19, False, (95, 102, 115))
slide.notes_slide.notes_text_frame.text = 'Title slide for the five-section CrackLab research presentation.'

# Move the newly appended slide before the original five slides.
slide_ids = prs.slides._sldIdLst
new_id = slide_ids[-1]
slide_ids.remove(new_id)
slide_ids.insert(0, new_id)
prs.save(deck_path)
assert len(Presentation(deck_path).slides) == 6
print(deck_path)
