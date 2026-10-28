"""Download and verify the public TouchAnnotator release and upstream models."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = 'RammusLeo/TouchAnnotator'
MODEL_REVISION = 'v1.0'


def download_student(destination, revision):
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ['config.yaml', 'manifest.json', 'model.pth']:
        hf_hub_download(MODEL_REPO, filename=filename, revision=revision,
                        local_dir=destination, token=False)
    manifest = json.loads((destination / 'manifest.json').read_text())
    checkpoint = destination / 'model.pth'
    if checkpoint.stat().st_size != manifest['bytes']:
        raise RuntimeError('TouchAnnotator checkpoint size mismatch')
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['sha256']:
        raise RuntimeError('TouchAnnotator checkpoint SHA256 mismatch')
    print('TouchAnnotator checkpoint SHA256 verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-model', action='store_true',
                        help='Download only the student checkpoint/config/manifest')
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Repository root in which assets will be placed')
    parser.add_argument('--model-revision', default=MODEL_REVISION)
    parser.add_argument('--qwen-revision', default='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    parser.add_argument('--wilor-revision', default='99fe3d7acff8104ecca1055df7467709506c2fa6')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    wilor = root / 'third_party/WiLoR'
    if not args.only_model and not (wilor / 'wilor/models/__init__.py').is_file():
        parser.error('Clone WiLoR into third_party/WiLoR first; see README step 1')
    download_student(root / 'checkpoints/touchannotator', args.model_revision)
    if not args.only_model:
        snapshot_download('Qwen/Qwen3.5-4B', revision=args.qwen_revision,
                          local_dir=root / 'checkpoints/Qwen3.5-4B', token=False)
        for filename in ['detector.pt', 'wilor_final.ckpt']:
            hf_hub_download('rolpotamias/WiLoR', repo_type='space',
                            filename=f'pretrained_models/{filename}',
                            revision=args.wilor_revision, local_dir=wilor, token=False)
    print('Assets ready. Run python scripts/check_assets.py', flush=True)


if __name__ == '__main__':
    main()
