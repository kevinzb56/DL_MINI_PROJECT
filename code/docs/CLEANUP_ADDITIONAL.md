# Additional cleanup

Run from the project root. No files are deleted without `-Delete`.

```powershell
.\scripts\operations\cleanup-additional.ps1
.\scripts\operations\cleanup-additional.ps1 -Delete
```

Default scope: only the old `artifacts/phase8/CrackLab_RC1.zip` plus its checksum.
The Python environment `.venv-multisource` and `web/node_modules` are preserved,
even though they occur in the older cleanup manifest. The final release ZIP is
retained and hash-checked. Original datasets, Multi-Source prepared packages,
model recovery ZIPs/checksums, and reports are outside the deletion allowlist.

Optional: delete the prepared Milano train/val/test copies, retaining all
original publisher files and the frozen manifest, audit and review image:

```powershell
.\scripts\operations\cleanup-additional.ps1 -IncludePreparedMilano
.\scripts\operations\cleanup-additional.ps1 -IncludePreparedMilano -Delete
```

This option disables local Milano data use until rebuilt. Every original image
and raw mask is verified against the frozen preparation manifest before deletion.
Rebuild into a new directory because the preparation script rejects existing
output directories:

```powershell
python scripts/data/prepare_milano.py --output data/external/milano_prepared_rebuilt
```

Use an environment with NumPy and Pillow installed. After successful preparation,
compare the rebuilt manifest's records with the retained manifest and restore the
rebuilt train/val/test directories into `milano_prepared_v1`. Keep the original
metadata. Preparation requires temporary disk headroom.

The script checks the exact reviewed file inventory and SHA-256 hashes and stops
if contents changed. It excludes datasets other than the optional derived copy,
Git history, research results, current checkpoints, and inaccessible test folders.
It does not invoke the earlier cleanup script. Execution results are written to
`output/cleanup_additional_execution_*.json`.
