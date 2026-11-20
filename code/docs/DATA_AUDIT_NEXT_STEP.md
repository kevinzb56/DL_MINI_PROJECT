# Real-data audit handoff

The imported Phase 1 ZIP contains dataset commit IDs but no raw image/mask pairs.
Direct agent network access remains unavailable. To audit the actual data, run
`output/kaggle/Road_Crack_Phase1_Data_Audit.ipynb` on the already-connected Kaggle
kernel. It embeds the new script and uses the previous data directory.

The notebook creates `phase1_data_audit.zip` in `/kaggle/working`. Download through
Kaggle's Files/Output panel and put it in this workspace. It contains:

- Measured counts, annotation values, dimensions, hashes, and foreground fractions.
- Pairing/annotation errors and full path inventory for diagnosing unexpected layouts.
- Exact decoded duplicate and dHash similarity findings.
- Derived binary masks with explicit conversion provenance; originals stay unchanged.
- A contact sheet showing original images, masks and overlays.
- Candidate source train/validation/holdout and target acquisition/query allocations.

No final manifest is frozen yet. Similarity proxy groups cannot establish real
capture/sequence independence. Review image polarity, errors, candidate clusters,
and metadata availability before freezing a clearly labeled protocol.
