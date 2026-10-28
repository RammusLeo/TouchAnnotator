# Release validation

Validated on 2026-10-27 (Asia/Shanghai).

## Completed checks

- 10 regression tests passed: source-time interpolation, pose-gap handling, clip padding, and ordered two-clip history.
- Three real EgoTouch ego-video excerpts, each 50 source frames / 25 sampled frames / four model clips, completed WiLoR → student inference → MP4 rendering.
- A fold-paper excerpt detected the left hand in 21/25 sampled frames and the right hand in 20/25. Two additional gloved/occluded excerpts had very low detection coverage and exercised missing-pose behavior.
- The student checkpoint loaded with strict key/shape matching. Pressure and contact outputs have shape `(50, 2, 21, 21)`, are finite, and lie in `[0, 1]`.
- Thresholded contact excludes invalid anatomical cells. Each rendered MP4 decodes to 50 frames.
- Tested on one NVIDIA H200; peak allocated GPU memory was 10,258,576,896 bytes (about 9.55 GiB). This is not a verified minimum VRAM requirement for all videos or GPU architectures.
- PyTorch 2.7.1+cu126, torchvision 0.22.1, Transformers 5.3.0, Python 3.10; SDPA/BF16.
- The tested WiLoR Python source exactly matches upstream commit `fcb911312a38fa8badd30d9656a167485d61b8f9`. Tested WiLoR weights match the pinned official HF repository by SHA256.
- Source files passed Python compilation checks and were scanned for private machine paths.

See `validation_results.json` for per-case detection coverage, value ranges and memory measurements. Inputs and prediction videos are retained in the maintainer's private staging area; raw dataset clips are not included in the public source archive.

## Scope

Testing used an isolated virtual environment with the existing server PyTorch stack plus separately installed WiLoR dependencies. A completely empty-machine installation has not been tested. Dependency downloads on this server required an offline wheel transfer because its mirror returned errors; these are infrastructure issues, not a required runtime step for users with normal PyPI access.

This verifies executable inference and output contracts, not all paper metrics, model calibration, or accuracy on arbitrary user videos. Pressure is relative to training sensors. Severe occlusion, gloves, multiple people and missed WiLoR detections remain limitations.

The trained student, matching configuration and SHA256 manifest are versioned at https://huggingface.co/RammusLeo/TouchAnnotator, tag `v1.0`. The source bundle, model card and inference guide are available on the repository's `main` branch. The anonymous downloader obtains matching `model.pth`, `config.yaml` and `manifest.json` and verifies the checkpoint SHA256.

## Run the regression tests

```bash
python -m pip install pytest
python -m pytest tests -q
```
