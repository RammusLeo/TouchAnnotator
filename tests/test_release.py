from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/core'))
from inference_full_causal_video_pose import infer

class DummyModel:
    multi_view_enabled = True
    clip_memory = SimpleNamespace(history_clips=2, memory_tokens_per_clip=1, embed_dim=3)
    def __init__(self): self.history=[]
    def __call__(self, **kw):
        self.history.append(kw['history_valid'].tolist()[0])
        n = kw['poses'].shape[1]
        return {'tactile':torch.full((1,n,2,21,21),0.25),
                'contact':torch.full((1,n,2,21,21),0.75),
                'clip_memory':torch.ones(1,1,3)*len(self.history)}

def test_causal_history_and_partial_clip():
    model=DummyModel()
    cfg={'data':{'clip_length':8,'image_size':[16,16],'pose_coordinate_mode':'wrist_relative'}}
    frames=np.zeros((25,16,16,3),dtype=np.uint8)
    poses=np.zeros((25,21,3),dtype=np.float32)
    valid=np.ones(25,dtype=bool)
    pressure,contact=infer(model,cfg,frames,poses,poses,valid,valid,torch.device('cpu'))
    assert pressure.shape==contact.shape==(25,2,21,21)
    assert model.history==[[False,False],[False,True],[True,True],[True,True]]
    np.testing.assert_allclose(pressure,0.25)
    np.testing.assert_allclose(contact,0.75)
