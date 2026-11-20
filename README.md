# CrackLab workspace

The project is organized into three folders:

| Folder | Contents |
| --- | --- |
| [code](code/) | Application, research source, configuration, tests, scripts, and documentation |
| [data](data/) | Downloaded datasets, manifests, and bundled demo/replay inputs |
| [results](results/) | Experiment outputs, workspaces, model archives, reports, and submission ZIPs |

Open a terminal in `code/` to run the application:

```powershell
cd code
powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1
.\Start_CrackLab.cmd
```

For the current submission, use [the submission guide](code/SUBMISSION_README.md) and the ZIP in [results/submission](results/submission/). For research methods and results, see [the project README](code/README.md).

`code/data`, `code/output`, `code/artifacts`, and `code/release_assets` are Windows junctions to the corresponding folders under `data/` and `results/`. They preserve existing relative paths; the files are stored only in the three folders shown above.

Some historical run folders under `results/` contain the exact notebooks and remote scripts used for those runs. They remain with their logs and checkpoints as frozen evidence. Maintained source code is under `code/`.
