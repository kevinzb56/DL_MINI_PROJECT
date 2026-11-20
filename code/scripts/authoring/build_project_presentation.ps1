$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'output\presentations'
New-Item -ItemType Directory -Force $out | Out-Null
$app=New-Object -ComObject PowerPoint.Application
$deck=$app.Presentations.Add()
$deck.PageSetup.SlideWidth=960
$deck.PageSetup.SlideHeight=540
$navy=2432020; $teal=7891712; $gray=6513507
function Text($slide,$value,$x,$y,$w,$h,$size=22,$bold=$false,$color=$navy) {
 $s=$slide.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $s.TextFrame.WordWrap=-1
 $s.TextFrame.TextRange.Text=$value
 $s.TextFrame.TextRange.Font.Name='Aptos'
 $s.TextFrame.TextRange.Font.Size=$size
 $s.TextFrame.TextRange.Font.Bold=[int]$bold*-1
 $s.TextFrame.TextRange.Font.Color.RGB=$color
 return $s
}
function Slide($title,$body,$note) {
 $s=$deck.Slides.Add($deck.Slides.Count+1,12)
 $s.Background.Fill.ForeColor.RGB=16777215
 $null=Text $s $title 48 32 864 86 32 $true
 $null=Text $s $body 48 134 864 340 22
 $null=Text $s ('CRACKLAB   /   '+$deck.Slides.Count.ToString('00')) 48 502 864 20 10 $false $gray
 $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$note
 return $s
}
function TableSlide($title,$rows,$caption,$notes) {
 $s=Slide $title '' $notes
 $cols=$rows[0].Count
 $table=$s.Shapes.AddTable($rows.Count,$cols,48,140,864,280).Table
 for($r=1;$r -le $rows.Count;$r++) { for($c=1;$c -le $cols;$c++) {
  $cell=$table.Cell($r,$c)
  $cell.Shape.TextFrame.TextRange.Text=[string]$rows[$r-1][$c-1]
  $cell.Shape.TextFrame.TextRange.Font.Name='Aptos'
  $cell.Shape.TextFrame.TextRange.Font.Size=18
  $cell.Shape.TextFrame.TextRange.Font.Color.RGB=$navy
  $cell.Shape.Fill.ForeColor.RGB=if($r -eq 1){15198183}else{16777215}
  $cell.Shape.TextFrame.TextRange.Font.Bold=if($r -eq 1){-1}else{0}
 } }
 $null=Text $s $caption 48 438 864 55 17 $false $gray
}
$sources='Sources: MODEL_CARD.md; docs/FINAL_REPORT.md. Core results use one training seed, frozen source holdout n=237 and target query n=95. Query was inspected in earlier phases. Physical site independence is unverified.'
$null=Slide 'CrackLab' "Lightweight road crack segmentation`n`nBoundary-aware learning and few-shot adaptation`n`nCollege project review`nNovember 2026" $sources
$null=Slide 'The problem and project objectives' "Road texture, lighting and camera height change the appearance of cracks.`n`nWe predict crack pixels with a compact model and adapt it using a few labeled images.`n`nThe local application lets a reviewer inspect predictions, correct masks and validate model updates." $sources
TableSlide 'Datasets for the core experiment' @(@('Partition','Dataset','Pairs'),@('Source training','DeepCrack','240'),@('Source validation','DeepCrack','60'),@('Source holdout','DeepCrack','237'),@('Target acquisition','CrackMap','25'),@('Target query','CrackMap','95')) 'Frozen partitions separate support selection from query scoring.' $sources
TableSlide 'Expanded dataset: ground cameras and UAVs' @(@('Property','Multisource pavement dataset'),@('Image-mask pairs','3,564'),@('Train / validation / test','2,526 / 542 / 496'),@('Platforms','GoPro, iPhone and UAV'),@('UAV capture heights','3 to 100 m plus mixed groups'),@('Original data','Six archives, 14.68 GiB')) 'Separate experiment from the DeepCrack / CrackMap benchmark.' 'Source: https://zenodo.org/records/22697998 ; docs/DATASET_DOWNLOAD_LINKS.md. Version 1.0.0, CC BY 4.0. Authors: Chaudhry Abu Bakar Imran and Flavia-Ioana Patrascu. Official test scoring is deferred.'
$null=Slide 'Our approach' "1. Train a compact region segmenter on source images.`n`n2. Add boundary supervision to emphasize crack edges.`n`n3. Freeze the encoder and adapt decoder/output heads on support masks.`n`n4. Validate a candidate before explicit activation." ($sources+' Boundary supervision does not guarantee connected cracks.')
TableSlide 'LiteCrack model implementation' @(@('Component','Implementation'),@('Encoder','Half-width MobileNetV3-Small, random initialization'),@('Decoder','Multiscale skips, depthwise-separable blocks'),@('Normalization','GroupNorm'),@('Region / boundary objectives','BCE + Dice / 3 x 3 morphological gradient'),@('Parameters','339,265 LiteCrack / 339,282 LiteCrack-BA')) 'The boundary head adds 17 parameters.' ($sources+' Source code: src/litecrack.py; scripts/experiments/train_baseline.py.')
$null=Slide 'Training and evaluation' "Core benchmark: 256 x 256 RGB inputs, threshold 0.50.`n`nValidation selects checkpoints. Dice and IoU use per-image macro means.`n`nEach few-shot run starts from its original parent checkpoint.`n`nK = 1, 5 or 10 support images, five draws and five support passes. Compute changes with K." ($sources+' Bilinear RGB resize, nearest-neighbor masks. Two empty masks score 1. Adam 0.0001 for adaptation, batch two, encoder frozen.')
$null=Slide 'Local inspection application' "React interface with original-resolution mask editing`n`nFastAPI inference, SQLite metadata and file snapshots`n`nSeparate adaptation workers and checksummed best/latest recovery`n`nCandidate validation, explicit activation and rollback" ($sources+' Source: inspection/ and web/. Functional demo covers real dataset images, existing reference masks, adaptation and exports. No field deployment or expert correction-time claim.')
TableSlide 'Static model results' @(@('Model','Parameters','Source Dice','Target Dice'),@('U-Net','1,965,569','81.32%','24.60%'),@('R2AU reference','2,110,029','81.42%','34.52%'),@('LiteCrack','339,265','76.36%','34.01%'),@('LiteCrack-BA','339,282','75.40%','41.72%')) 'BA adds 7.71 target Dice points. LiteCrack uses 83.92% fewer parameters than our R2AU reference.' ($sources+' R2AU is a paper-inspired local implementation. Source accuracy decreases. No universal superiority claim.')
TableSlide 'Few-shot adaptation results' @(@('Support labels','LiteCrack Dice','LiteCrack-BA Dice'),@('0','34.01%','41.72%'),@('1','41.46 +/- 8.93%','48.00 +/- 6.90%'),@('5','54.85 +/- 1.07%','57.23 +/- 2.00%'),@('10','56.91 +/- 1.33%','59.09 +/- 1.76%')) 'Five-shot BA gains 15.51 target Dice points, with a 2.81-point source Dice loss.' ($sources+' Mean +/- sample SD over five support draws on the same query set. SD is not a confidence interval. Five/ten-shot BA misses the optional two-point source retention target; ten-shot source change -4.87 points.')
$null=Slide 'Recent multisource experiments' "LiteCrack, seed 42, batch 8, Adam 0.0001.`n`n256 px run: best validation Dice 23.86%, epoch 53. Training ended at epoch 68.`n`n512 px pilot: validation Dice 26.23% at epoch 20.`n`nBoth use 542 official validation images. Official test scoring remains deferred." 'Sources: docs/MULTISOURCE_FULL_TRAINING.md; docs/MULTISOURCE_512_PILOT.md; final recovery ZIPs in output/multisource_full_v1/recovery and output/multisource_512_pilot/recovery, artifacts/run/history.json and status.json. Unequal training horizons prevent a clean causal claim about resolution. These validation results are separate from the core target-query results. Milano preparation exists, but final Milano checkpoints were unavailable for the planned transfer baseline.'
TableSlide 'Efficiency and functional evidence' @(@('Evidence','Recorded result'),@('LiteCrack native CPU p95','44.88 ms'),@('LiteCrack-BA native CPU p95','52.59 ms'),@('ONNX probability parity','Maximum error below 0.001'),@('Local workflow','Prediction, editing, adaptation and rollback'),@('Fresh release tests','49 passed, 6 CUDA-only skips')) 'Model-only timing: i5-1335U, 256 x 256, batch one, two threads.' ($sources+' Timing uses 20 warmups and 200 timed passes. Excludes preprocessing, loading, UI and tiling. ONNX parity covers eight audited source-validation samples; no speed or global accuracy claim. Historical release tests, not rerun for this deck.')
$null=Slide 'Limitations and future scope' "Current limits`nSingle training seed, query reuse, vegetation false positives and source forgetting.`n`nNext experiments`nIndependent site-separated testing, repeated seeds, complete hard-negative masks and road-region filtering.`n`nEngineering direction`nStronger replay/distillation retention, measured correction effort and full inference benchmarks on target devices." ($sources+' Future work is proposed. No road-safety, physical crack-dimension or autonomous deployment claims.')
$null=Slide 'Project outcome' "A compact segmentation model with measurable target-domain gains.`n`nFew-shot adaptation improves pilot accuracy and exposes a retention tradeoff.`n`nA working application connects predictions, corrections and controlled model updates.`n`nNext milestone: independent evaluation and stronger source retention." ($sources+' References: Katsamenis et al., A Few-Shot Attention Recurrent Residual U-Net for Crack Segmentation, arXiv:2303.01582; Liu et al., DeepCrack, Neurocomputing 338 (2019), 139-153; Zenodo DOI 10.5281/zenodo.22697998.')
foreach($visual in @(@('Real-image prediction and reference mask','real_demo.png','Known dataset image and existing reference annotation. This demonstrates functionality, not independent accuracy.'),@('Broad-scene failure case','wide_scene_failure.png','Green shows model predictions. Vegetation and scene edges still cause false positives. No full reference mask exists for this photo.'))) {
 $s=Slide $visual[0] '' ($sources+' Figure source: output/phase8/figures/'+$visual[1])
 $pic=$s.Shapes.AddPicture((Join-Path $root ('output/phase8/figures/'+$visual[1])),0,-1,48,128,-1,-1)
 $pic.LockAspectRatio=-1
 if($pic.Width/864 -gt $pic.Height/300){$pic.Width=864}else{$pic.Height=300}
 $pic.Left=(960-$pic.Width)/2
 $null=Text $s $visual[2] 48 442 864 54 17 $false $gray
 $s.MoveTo(9)
}
$cover=$deck.Slides.Item(1)
$cover.Background.Fill.ForeColor.RGB=2432020
foreach($shape in $cover.Shapes){if($shape.HasTextFrame){$shape.TextFrame.TextRange.Font.Color.RGB=16777215}}
$cover.Shapes.Item(1).TextFrame.TextRange.Font.Size=54
$cover.Shapes.Item(1).Top=64
for($i=2;$i -le $deck.Slides.Count;$i++){$deck.Slides.Item($i).Shapes.Item(1).TextFrame.TextRange.Font.Color.RGB=7891712}
$file=Join-Path $out 'CrackLab_Project_Review.pptx'
$deck.SaveAs($file,24)
$renders=Join-Path $out 'slide_previews'
New-Item -ItemType Directory -Force $renders | Out-Null
$deck.Export($renders,'PNG',1280,720)
$deck.SaveAs((Join-Path $out 'CrackLab_Project_Review.pdf'),32)
$deck.Close()
$app.Quit()
Write-Output $file
