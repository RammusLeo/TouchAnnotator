# Inference reference

[← Project overview](../README.md)

Run all commands from the repository root.

## Outputs

| File | Contents |
| --- | --- |
| `prediction.mp4` | Input video beside left/right relative-pressure maps |
| `predictions.npz` | Dense pressure, contact probability, contact decisions, poses and timing |
| `wilor_poses.jsonl` | Sampled source-frame index, camera-frame MANO-21 joints, detections, gap filling and validity |
| `metadata.json` | Frame counts, FPS, pose coverage, numerical settings and GPU memory |

```python
import numpy as np
x = np.load('outputs/my_video/predictions.npz')
pressure = x['pressure']                       # (N, 2, 21, 21), [0, 1]
contact = x['contact_probability']             # (N, 2, 21, 21), [0, 1]
valid_cells = x['valid_cells']                 # (2, 21, 21), anatomical support
hand_contact = x['hand_contact']               # (N, 2), thresholded valid cells
indices = x['sampled_source_indices']          # (S,), indexes into source frames
left_pose = x['left_pose']                     # (S, 21, 3), camera frame, metres
left_valid = x['left_valid']                   # (S,), validity after short-gap filling
```

Hand axis order is **left, right**. Source-rate fields have `N` frames; pose arrays and `sampled_*` fields have `S` sampled frames. Use `sampled_source_indices` to align them. The default threshold is 0.5; it is a decision threshold, not a calibrated contact guarantee. `contact_state` and `hand_contact` exclude cells outside `valid_cells`; dense model outputs retain all grid cells.

## Preprocessing and interpretation

- RGB is resized to 224 × 224 and normalized by mean/std 0.5. Sampling targets 15 FPS using an integer source-frame interval; actual sampling FPS can differ for non-divisible source rates.
- Eight-frame clips are processed in order, with two preceding clips compressed to 16 tokens each. The final partial clip repeats its last frame and trims padded predictions.
- The current clip is reasoned over bidirectionally. Memory is past-only across clips; this is not frame-online inference. Source-rate outputs are linearly interpolated from sampled predictions.
- WiLoR estimates up to one left/right hand pair. Use a first-person video of one person's hands; multiple people, severe occlusion, motion blur and off-screen hands can cause errors.
- Gaps of at most three sampled frames are filled and poses smoothed with radius one. These operations may use later sampled poses; the full video preprocessing is offline. Long missing detections remain invalid. A predicted field for a missing hand should not be interpreted as an observation of that hand.
- Pressure is sensor-relative, normalized to [0,1], and has no calibrated physical unit such as Newtons or kPa. Contact and pressure are separate outputs.
- The model's GPU history is bounded, but this initial demo retains decoded sampled frames and prediction arrays in host RAM. Split long videos into manageable files; each invocation resets memory.

## Reproducibility and scope

The supplied checkpoint is the student `best_model.pth` from run `20260824_003600_qwen3.5_4b_tactile_prediction_qwen35_full_causal_memory_opd_full_qwen`, referenced by the existing Full/FPHA evaluation scripts. Export preserves original tensor values and dtypes and removes optimizer, scheduler, and EMA teacher state. Loading is strict. The portable config retains model architecture and inference preprocessing, removes dataset paths, and switches the reasoner's attention implementation to SDPA. This demo is not a claim to reproduce every paper table; those require the original datasets, splits and evaluation protocols.

The checkpoint exporter and checker use tensor-only loading and SHA256. Only load model assets obtained from the documented upstreams or a trusted release.

## Acknowledgements and license

This implementation builds on TouchAnything, [WiLoR](https://github.com/rolpotamias/WiLoR), and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Original project code retains its [MIT license](../LICENSE); third-party models retain their own terms. See [THIRD_PARTY.md](../THIRD_PARTY.md).


