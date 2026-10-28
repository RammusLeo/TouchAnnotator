"""Export student tensors only, preserving original precision and values."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
a=p.parse_args()
if a.destination.exists(): raise SystemExit(f'Refusing overwrite: {a.destination}')
x=torch.load(a.source,map_location='cpu',weights_only=True,mmap=True)
state=x['model_state_dict']
a.destination.parent.mkdir(parents=True,exist_ok=True)
torch.save({'model_state_dict':state},a.destination)
h=hashlib.sha256()
with a.destination.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
manifest={'file':a.destination.name,'sha256':h.hexdigest(),'bytes':a.destination.stat().st_size,
 'tensors':len(state),'source_run':a.source.parent.name,'epoch':x.get('epoch'),
 'contents':'student model_state_dict; original dtype; no optimizer or teacher'}
(a.destination.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
