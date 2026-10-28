<div align="center">

<img src="assets/brand/logo.png" alt="TouchAnnotator" width="800">

<h3>多源触觉监督与特权视角蒸馏</h3>
<p><strong>从一段第一人称视频，估计双手的相对压力与接触分布。</strong></p>

<img src="assets/brand/capabilities.png" alt="单路 ego 输入 · WiLoR 姿态 · 压力与接触 · 历史片段记忆" width="760">

<p><a href="https://github.com/RammusLeo/TouchAnnotator">💻 源码</a> · <a href="https://huggingface.co/RammusLeo/TouchAnnotator">🤗 模型与权重</a> · <a href="README.md">English</a> · <a href="docs/INFERENCE.md">📖 完整推理指南</a> · <a href="VALIDATION.md">🔬 验证记录</a></p>

<img src="assets/paper/overview.png" alt="论文中的 TouchAnnotator 总览" width="100%">
<sub>论文图 1：概念示意。热图为示意图，不代表实际预测或传感器测量。</sub>

</div>

## ✨ 一段视频，两只手，两类触觉输出

输入单路第一人称 RGB 视频，先用 **WiLoR** 估计双手姿态，再预测每只手的 **21×21 相对压力图与接触概率图**。推理不需要腕部相机或触觉传感器。

| 🤲 双手预测 | 🎥 单路视频部署 | 🧩 多源训练 |
| :--- | :--- | :--- |
| 左右手分别输出压力与接触概率 | 从同一视频估计姿态并预测触觉 | 整合手套、裸手与灵巧机械手监督 |

<img src="assets/paper/egotouch-results.png" alt="EgoTouch 触觉预测对比" width="100%">

*论文图 3 上半部分：ego-only TouchAnnotator 与多视角 TouchAnything、测量真值的定性比较。*

完整安装步骤、权重准备和参数说明见 [英文 README](README.md)。

## 🚀 快速开始

```bash
git clone https://github.com/RammusLeo/TouchAnnotator.git
cd TouchAnnotator
```


1. 按英文 README 创建 Python 3.10 + CUDA 环境并安装 requirements.txt。
2. 检出英文 README 指定版本的 WiLoR。
3. 运行 `python scripts/download_assets.py`，一次下载 TouchAnnotator 学生权重、配套配置、Qwen 与 WiLoR 模型，并校验学生权重的 SHA256。
4. 运行 `python scripts/check_assets.py` 检查完整资产。
5. 执行：

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --video /path/to/bimanual_video.mp4 \
  --output-dir outputs/my_video
```

首次测试可加 `--max-frames 50`；不需要预览视频时加 `--no-video`。

🤗 [公开模型仓库](https://huggingface.co/RammusLeo/TouchAnnotator) · [权重直链（9.40 GB）](https://huggingface.co/RammusLeo/TouchAnnotator/resolve/v1.0/model.pth) · [源码包](https://huggingface.co/RammusLeo/TouchAnnotator/resolve/main/TouchAnnotator-source.zip)

一键下载全部模型资产，无需登录：

```bash
python scripts/download_assets.py
```

国内网络可选择下载镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_assets.py
```

下载脚本使用匿名请求，无需传入 Hugging Face Token；[HF-Mirror 用法](https://hf-mirror.com)。

已有 Qwen 和 WiLoR 时，仅下载 TouchAnnotator：

```bash
python scripts/download_assets.py --only-model
```

等价的 Hugging Face CLI 命令：

```bash
hf download RammusLeo/TouchAnnotator model.pth config.yaml manifest.json --revision v1.0 --local-dir checkpoints/touchannotator
```

## 📦 输出结果

- `prediction.mp4`：原视频和双手压力热图。
- `predictions.npz`：压力、接触概率、接触状态、时间索引、姿态和有效标记。
- `wilor_poses.jsonl`：采样帧的 WiLoR 姿态及原始检测、短缺口补全标记。
- `metadata.json`：帧数、检测覆盖率、运行精度和峰值显存。

手的顺序为左、右。压力是归一化的传感器相对量，不是牛顿或 kPa。双手姿态只在采样帧输出，使用 `sampled_source_indices` 与原视频帧对齐。手未被检出时模型仍会输出预测，应结合有效标记判断，不能当作已观测到该手。

本次包含 Full 学生模型的推理权重，约 9.40 GB，保留原始参数精度，去掉优化器和 EMA 教师状态。权重、配置与校验清单在 Hugging Face `v1.0` 版本中配套提供，无需申请访问。推理采用 SDPA/BF16，与训练配置的 FlashAttention 后端不同；不声称复现所有论文表格。

## 🧠 方法概览

<img src="assets/paper/method.png" alt="多源监督、特权视角教师、ego-only 学生与历史片段记忆" width="100%">

*论文图 2。多视角 EMA 教师只在训练阶段提供指导；部署时学生使用 ego 视频与估计姿态。历史记忆保留前两个片段，每个片段压缩为 16 个 token。*

## 🎨 跨手型结果

<img src="assets/paper/cross-embodiment.png" alt="机械手、裸手与触觉手套的预测示例" width="100%">

*论文图 5。同类操作在灵巧机械手、裸手和触觉手套上的定性示例；不同传感器的颜色数值不代表统一物理压力。*

## 🔬 已验证的运行路径

10 项回归测试通过，3 段真实 ego 视频完成端到端推理。双手折纸片段的 25 个采样帧中，左手检出 21 帧、右手检出 20 帧。H200 上峰值分配显存约 10.26 GB；这不是其他硬件的最低显存承诺。详见 [验证记录](VALIDATION.md)。

所有论文插图均记录在 [素材来源说明](assets/README.md) 中。
