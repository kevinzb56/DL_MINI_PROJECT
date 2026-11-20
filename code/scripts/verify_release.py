"""Check bundle/extracted file integrity without importing ML or app packages."""
import argparse
import json
from pathlib import Path
from release_integrity import verify_archive,verify_directory

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(verify_archive(args.archive) if args.archive else verify_directory(args.root),indent=2))
