$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'output\presentations'
$app=New-Object -ComObject PowerPoint.Application
$p=$app.Presentations.Open((Join-Path $out 'CrackLab_Project_Review.pptx'),0,0,0)
function Txt($s,$v,$x,$y,$w,$h,$size=20,$bold=$false) {
 $t=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $t.TextFrame.TextRange.Text=$v
 $t.TextFrame.TextRange.Font.Name='Aptos'
 $t.TextFrame.TextRange.Font.Size=$size
 $t.TextFrame.TextRange.Font.Bold=if($bold){-1}else{0}
 $t.TextFrame.TextRange.Font.Color.RGB=2432020
 return $t
}
function Find($title){foreach($s in $p.Slides){if($s.Shapes.Item(1).TextFrame.TextRange.Text -eq $title){return $s}};throw "Missing slide $title"}
# Combine the two datasets into one flat comparison.
$s=Find 'Datasets for the core experiment'
for($i=$s.Shapes.Count;$i -ge 2;$i--){$s.Shapes.Item($i).Delete()}
$s.Shapes.Item(1).TextFrame.TextRange.Text='Datasets and evaluation splits'
$null=Txt $s 'CORE FEW-SHOT EXPERIMENT' 48 140 410 32 19 $true
$null=Txt $s "DeepCrack source`n240 training / 60 validation / 237 holdout`n`nCrackMap target`n25 acquisition / 95 query pairs" 48 190 410 230 21
$null=Txt $s 'MULTISOURCE EXPANSION' 500 140 410 32 19 $true
$null=Txt $s "3,564 image-mask pairs`n2,526 train / 542 validation / 496 test`n`nGoPro, iPhone and UAV at 3-100 m`nZenodo 22697998, CC BY 4.0" 500 190 410 230 21
$null=Txt $s 'These are separate experiments. Frozen splits support reproducibility; physical site independence remains unverified.' 48 438 864 55 17
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text+=' Multisource source: https://zenodo.org/records/22697998 . Version 1.0.0. Official multisource test scoring deferred.'
# Combine method, training and model implementation.
$s=Find 'Training and evaluation'
$s.Shapes.Item(1).TextFrame.TextRange.Text='Approach and training protocol'
$s.Shapes.Item(2).TextFrame.TextRange.Text="Train LiteCrack with BCE + Dice and boundary supervision.`n`nCore benchmark: 256 x 256 RGB, threshold 0.50, validation-selected checkpoints and per-image macro Dice/IoU.`n`nFreeze the encoder and adapt decoder/heads from 1, 5 or 10 support images.`n`nFive draws, five support passes each. Every run starts from its original parent. Compute varies with K."
$s.Shapes.Item(2).TextFrame.TextRange.Font.Size=21
# Add the requested editable diagram in place of the implementation table.
$s=Find 'LiteCrack model implementation'
for($i=$s.Shapes.Count;$i -ge 2;$i--){$s.Shapes.Item($i).Delete()}
$s.Shapes.Item(1).TextFrame.TextRange.Text='Model and application architecture'
function Box($s,$text,$x,$y,$w,$h){
 $b=$s.Shapes.AddShape(1,$x,$y,$w,$h)
 $b.Fill.ForeColor.RGB=15987699
 $b.Line.ForeColor.RGB=7891712
 $b.TextFrame.TextRange.Text=$text
 $b.TextFrame.TextRange.Font.Name='Aptos'
 $b.TextFrame.TextRange.Font.Size=16
 $b.TextFrame.TextRange.Font.Color.RGB=2432020
 $b.TextFrame.TextRange.ParagraphFormat.Alignment=2
 $b.TextFrame.VerticalAnchor=3
 return $b
}
function Arrow($s,$x1,$y1,$x2,$y2){$l=$s.Shapes.AddConnector(1,$x1,$y1,$x2,$y2);$l.Line.ForeColor.RGB=7891712;$l.Line.Weight=2;$l.Line.EndArrowheadStyle=3}
$null=Txt $s 'SEGMENTATION MODEL' 48 118 864 28 17 $true
$null=Box $s "RGB image`n256 x 256" 48 158 135 64
$null=Box $s "MobileNetV3-Small`nencoder, half width" 221 158 194 64
$null=Box $s "Multiscale decoder`nGroupNorm + separable blocks" 453 158 230 64
$null=Box $s "Region mask`nBoundary head (BA)" 721 158 191 64
Arrow $s 183 190 221 190
Arrow $s 415 190 453 190
Arrow $s 683 190 721 190
$null=Txt $s 'Skip connections link encoder features to the decoder. The BA head adds boundary supervision during training.' 48 235 864 42 16
$null=Txt $s 'LOCAL APPLICATION' 48 294 864 28 17 $true
$null=Box $s "React UI`nUpload / edit masks" 48 335 190 70
$null=Box $s "FastAPI`nInference / job control" 280 335 190 70
$null=Box $s "Isolated worker`nFew-shot adaptation" 512 335 190 70
$null=Box $s "SQLite + files`nMasks / checkpoints" 744 335 168 70
Arrow $s 238 370 280 370
Arrow $s 470 370 512 370
Arrow $s 702 370 744 370
$null=Txt $s '339,265 parameters (LiteCrack), 339,282 (BA). Encoder stays frozen during adaptation. Candidates require validation before activation.' 48 442 864 55 17
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='Architecture sources: src/litecrack.py; inspection/api.py; inspection/engine.py; inspection/store.py; scripts/inspection_worker.py; web/src/. Diagram is a logical component view, not every API/data connection. Random encoder initialization, depthwise-separable decoder, GroupNorm. Morphological gradient boundary target, BCE + Dice region loss. Activation and rollback remain explicit.'
# Combine the demo and efficiency evidence on one slide.
$s=Find 'Real-image prediction and reference mask'
$s.Shapes.Item(1).TextFrame.TextRange.Text='Implementation and demonstrated results'
foreach($sh in $s.Shapes){if($sh.Type -eq 13){$sh.Width=650;$sh.Left=155;$sh.Top=130}}
foreach($sh in $s.Shapes){if($sh.HasTextFrame -and $sh.Top -ge 438 -and $sh.Top -lt 500){$sh.TextFrame.TextRange.Text="React + FastAPI + SQLite. Native model-only CPU p95: 44.88 ms / 52.59 ms (LiteCrack / BA). Prediction, correction, adaptation and rollback demonstrated.";$sh.TextFrame.TextRange.Font.Size=17}}
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text+=' Efficiency: docs/FINAL_REPORT.md. i5-1335U, batch one, 256 x 256, two threads, 20 warmups, 200 passes. Excludes loading/preprocessing/UI. ONNX parity error below 0.001 on eight audited validation samples. Fresh release tests: 49 passed, six CUDA-only skips. Existing reference masks, no annotation-effort study.'
$keep=@('CrackLab','The problem and project objectives','Datasets and evaluation splits','Model and application architecture','Approach and training protocol','Static model results','Few-shot adaptation results','Recent multisource experiments','Implementation and demonstrated results','Limitations and future scope')
for($i=$p.Slides.Count;$i -ge 1;$i--){if($keep -notcontains $p.Slides.Item($i).Shapes.Item(1).TextFrame.TextRange.Text){$p.Slides.Item($i).Delete()}}
for($i=0;$i -lt $keep.Count;$i++){(Find $keep[$i]).MoveTo($i+1)}
for($i=1;$i -le $p.Slides.Count;$i++){
 $s=$p.Slides.Item($i)
 foreach($sh in $s.Shapes){if($sh.HasTextFrame -and $sh.TextFrame.TextRange.Text -like 'CRACKLAB   /*'){$sh.Delete();break}}
 $f=Txt $s ('CRACKLAB   /   '+$i.ToString('00')) 48 502 864 20 10
 if($i -eq 1){$f.TextFrame.TextRange.Font.Color.RGB=16777215}
}
if($p.Slides.Count -ne 10){throw 'Expected 10 slides'}
$file=Join-Path $out 'CrackLab_Project_Review_10_Slides.pptx'
$p.SaveAs($file,24)
$renders=Join-Path $out 'previews_10_slides'
New-Item -ItemType Directory -Force $renders | Out-Null
$p.Export($renders,'PNG',1280,720)
$p.SaveAs((Join-Path $out 'CrackLab_Project_Review_10_Slides.pdf'),32)
$p.Close()
$app.Quit()
Write-Output $file
