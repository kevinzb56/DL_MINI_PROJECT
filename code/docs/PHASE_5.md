# Phase 5 few-shot adaptation protocol

Implementation prepared; remote recovery tests and experiment results pending.
Run `output/kaggle/Road_Crack_Phase5_FewShot.ipynb`, then preserve/download
`phase5_results.zip`. If needed, attach both verified Phase 3 and Phase 4 backups.

Thirty independently initialized adaptations: two frozen parent models × five
frozen draws × K=1/5/10. Each run resets from its original parent; K=10 never starts
from the K=5 adapted model. Only named target acquisition masks enter training.
These are existing dataset labels; actual expert correction time is not measured.

Encoder frozen; decoder, region head and optional boundary head updated. Adam
0.0001, five support passes, batch two, clipping one, BCE+Dice plus the existing
0.2 boundary objective where applicable. No replay or augmentation. This means
5/15/25 optimizer steps for K=1/5/10. Different step counts are declared rather
than confused with an equal-compute experiment. No support/epoch/configuration
selection uses query performance; final fixed-budget checkpoint is evaluated.

Source validation, source holdout and fixed target query are measured after all
adaptations. K=0 parent scores provide paired references. Report mean and spread
across draws, every negative result, and source forgetting separately. Draws reuse
the same query images, so they are not independent new test samples.

Every complete optimizer step saves immutable recovery state, parent identity,
support IDs, seed, sampler order/cursor, optimizer, RNG and history. Exact encoder
equality is checked. CPU/CUDA interruption-resume tests and a query-support
rejection test gate execution. Original parent checkpoints are never overwritten.

Export includes latest snapshot for every run; historical step files remain in
Kaggle without automatic pruning. Back up outside the live session. The original
single-seed and previously inspected target-query limitations still apply.
