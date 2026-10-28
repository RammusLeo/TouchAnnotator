"""Check required assets before starting expensive GPU inference."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
required=['checkpoints/touchannotator/manifest.json','checkpoints/touchannotator/model.pth','checkpoints/touchannotator/config.yaml',
 'checkpoints/Qwen3.5-4B/config.json','checkpoints/Qwen3.5-4B/tokenizer.json',
 'third_party/WiLoR/wilor/models/__init__.py','third_party/WiLoR/pretrained_models/model_config.yaml',
 'third_party/WiLoR/pretrained_models/detector.pt','third_party/WiLoR/pretrained_models/wilor_final.ckpt',
 'third_party/WiLoR/mano_data/mano_mean_params.npz']
missing=[f for f in required if not (ROOT/f).is_file()]
if not list((ROOT/'checkpoints/Qwen3.5-4B').glob('*.safetensors')):missing.append('checkpoints/Qwen3.5-4B/*.safetensors')
if missing:sys.exit('Missing assets:\n'+'\n'.join(missing))
m=ROOT/'checkpoints/touchannotator/manifest.json'
if m.exists():
 manifest=json.loads(m.read_text()); h=hashlib.sha256()
 with (m.parent/manifest['file']).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=manifest['sha256']:sys.exit('TouchAnnotator checkpoint SHA256 mismatch')
 print('TouchAnnotator SHA256 verified')
for filename,expected in {
 'wilor_final.ckpt':'3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2',
 'detector.pt':'5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033',
}.items():
 h=hashlib.sha256()
 with (ROOT/'third_party/WiLoR/pretrained_models'/filename).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=expected:sys.exit(f'WiLoR asset SHA256 mismatch: {filename}')
index=ROOT/'checkpoints/Qwen3.5-4B/model.safetensors.index.json'
if index.exists():
 missing=[s for s in set(json.loads(index.read_text())['weight_map'].values()) if not (index.parent/s).is_file()]
 if missing:sys.exit(f'Missing Qwen shards: {missing}')
print('Required asset paths and WiLoR SHA256 checks passed')
