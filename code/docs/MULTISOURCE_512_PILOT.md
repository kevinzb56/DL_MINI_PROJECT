# Controlled 512px pilot

The training-only resolution audit sampled 24 evenly spaced masks per archive
(144 total). Nearest downsample/upsample Dice against original masks improves
from 0.754 to 0.875 for high UAV and 0.675 to 0.826 for other/mixed imagery when
moving from 256 to 512 pixels. See `output/multisource_resolution/audit.json`.
This measures label fidelity, not model accuracy or causation.

Prepare a separate random-initialized LiteCrack seed-42 pilot: 512px, batch 8,
20 epochs maximum, patience 15, Adam 0.0001, BCE+Dice, clipping 1.0, same official
2,526/542 train/validation rows, shuffled across archives. Test scoring disabled.
Compare the fixed epoch-20 result and domain metrics with the original 256px
epoch-20 checkpoint/history, rather than claiming superiority from unequal
training horizons. Record compute cost and any early stopping. The 68-epoch
baseline remains immutable. No threshold or sampling changes are introduced.

Inputs are rebuilt from original verified archive members, not upscaled from
the 256px cache. Local builder: `build_multisource_full.py --size 512 --epochs 20
--output output/multisource_512_pilot/Training_512_Pilot.zip`.
The launcher waits for a complete checksummed package, uploads, verifies and
starts a distinct remote project, then synchronizes best/latest recovery locally.
Inspect `output/multisource_512_pilot/launch_status.json` before any restart.
Credentials are held in process memory only.
