#!/usr/bin/env python3
"""Annotate an egocentric bimanual video with relative pressure and contact."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/demo'))
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/touchannotator/model.pth')
    p.add_argument('--config', type=Path, default=ROOT/'checkpoints/touchannotator/config.yaml')
    p.add_argument('--wilor-dir', type=Path, default=ROOT/'third_party/WiLoR')
    p.add_argument('--mano-dir', type=Path)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--contact-threshold', type=float, default=0.5)
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0: p.error('--max-frames must be positive')
    if not 0 < args.contact_threshold < 1: p.error('--contact-threshold must be between 0 and 1')
    for f in [args.video,args.checkpoint,args.config]:
        if not f.is_file(): p.error(f'Missing file: {f}')
    import gc
    import json
    import cv2
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(ROOT/'scripts/core'))
    import inference_from_video_wilor as common
    import inference_full_causal_video_pose as causal
    from src.data.tactile_preprocessing import resolve_frame_interval
    from src.inference.video_pipeline import interpolate_at_positions, stabilize_pose_sequence
    if not torch.cuda.is_available(): p.error('CUDA GPU required for this release')
    if not args.no_video:
        import shutil
        if not shutil.which('ffmpeg'): p.error('Install ffmpeg or use --no-video')
    device = torch.device(f'cuda:{args.gpu_id}')
    cfg = yaml.safe_load(args.config.read_text())
    cap = cv2.VideoCapture(str(args.video)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    interval = resolve_frame_interval(fps,cfg['data'].get('target_fps'),cfg['data'].get('frame_interval',1))
    bgr,indices,n,fps = common.read_sampled_video(args.video,interval,args.max_frames)
    out = args.output_dir.resolve()
    if (out/'predictions.npz').exists(): p.error(f'Output already exists: {out}; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    wd = args.wilor_dir.resolve()
    estimator = common.WiLoREstimator(wd,wd/'pretrained_models/wilor_final.ckpt',wd/'pretrained_models/detector.pt',(args.mano_dir or wd/'mano_data').resolve(),device)
    raw = estimator.estimate(bgr,detector_batch_size=8,wilor_batch_size=16,confidence=0.3,rescale_factor=2.0)
    poses={}
    for side in ['left','right']:
        pose,valid,filled=stabilize_pose_sequence(raw[side],raw[side+'_valid'],max_gap=3,smooth_radius=1)
        poses.update({side:pose,side+'_valid':valid,side+'_filled':filled,side+'_detected':raw[side+'_valid'],side+'_score':raw[side+'_score']})
    common.save_pose_jsonl(out/'wilor_poses.jsonl',indices,poses)
    del estimator; gc.collect(); torch.cuda.empty_cache()
    model,cfg=causal.load_model(args.config.resolve(),args.checkpoint.resolve(),device)
    rgb=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in bgr])
    pressure,contact=causal.infer(model,cfg,rgb,poses['left'],poses['right'],poses['left_valid'],poses['right_valid'],device)
    for name,a in [('pressure',pressure),('contact',contact)]:
        if a.shape != (len(indices),2,21,21) or not np.isfinite(a).all():
            raise RuntimeError(f'Invalid {name} prediction: shape={a.shape}, finite={np.isfinite(a).all()}')
    src_times=np.arange(n,dtype=np.float64)/fps
    sample_times=indices.astype(np.float64)/fps
    full_pressure=interpolate_at_positions(pressure,sample_times,src_times)
    full_contact=interpolate_at_positions(contact,sample_times,src_times)
    support=np.stack([~common.SENSOR_NAN_MASK_L,~common.SENSOR_NAN_MASK_R])
    state=(full_contact>=args.contact_threshold)&support[None]
    np.savez_compressed(out/'predictions.npz',pressure=full_pressure,contact_probability=full_contact,
        contact_state=state,hand_contact=state.reshape(n,2,-1).any(-1),valid_cells=support,
        sampled_pressure=pressure,sampled_contact_probability=contact,sampled_source_indices=indices,
        left_pose=poses['left'],right_pose=poses['right'],left_valid=poses['left_valid'],right_valid=poses['right_valid'],
        source_fps=np.float32(fps),timestamps=src_times,contact_threshold=np.float32(args.contact_threshold))
    metadata={'source_frames':n,'sampled_frames':len(indices),'source_fps':fps,'frame_interval':interval,
      'left_detected':int(raw['left_valid'].sum()),'right_detected':int(raw['right_valid'].sum()),
      'pressure_units':'sensor-relative, normalized [0,1]','precision':'bfloat16','attention':'sdpa',
      'torch':torch.__version__,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device)}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if not args.no_video: common.render_prediction_video(args.video.resolve(),out/'prediction.mp4',full_pressure,fps,args.max_frames)
    print(f'Saved predictions to {out}')

if __name__=='__main__': main()
