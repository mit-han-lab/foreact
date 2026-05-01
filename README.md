# ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

### [Paper](http://arxiv.org/abs/2602.12322) | [Model](https://huggingface.co/mit-han-lab/foreact-pretrained) | [Real-World Data](https://huggingface.co/datasets/mit-han-lab/ForeActDataset)


## News

- \[2026/04\] 🔥 We make a [coffee making demo](#demo).
- \[2026/04\] 🔥 We support multi-view visual foresight generation with per-view customizable resolution.
- \[2026/02\] We release the code and model for ForeAct!


## About

- ForeAct is a **visual foresight planner** that empowers VLAs with the ability to **anticipate future observations**, enabling more informed decision-making.
- ForeAct is **general** and **plug-and-play**: state-of-the-art VLAs can seamlessly incorporate ForeAct **without any architectural modification**.
- ForeAct is **highly efficient**, generating a high-fidelity **640 $\times$ 480** future observation in just **0.33s** on a single H100 GPU.

<p align="left">
    <img src="assets/teaser.png">
</p>


## Demo

[![Watch the video](assets/coffee-demo-first-frame.png)](https://www.youtube.com/watch?v=rKdaN5hNqUs)

[![Watch the video](assets/demo-first-frame.png)](https://www.youtube.com/watch?v=D7fZT01Kd1A)

## Usage


### Environment Setup

```bash
git clone https://github.com/mit-han-lab/foreact
cd foreact
bash environment_setup.sh foreact
```

### Finetune

Download the [pretrained weights](https://huggingface.co/mit-han-lab/foreact-pretrained) and prepare your own real-world data (or use our processed [real-world data](https://huggingface.co/datasets/mit-han-lab/ForeActDataset)). Update the relevant paths in the config you want to run (`configs/finetune.yaml` for single-view, `configs/finetune_multiview.yaml` for multi-view), then launch:

```bash
### Single-view
bash scripts/run_finetune.sh finetune.yaml

### Multi-view
bash scripts/run_finetune.sh finetune_multiview.yaml
```

### Inference

```bash
### CLI (single-view)
python app_cli.py --checkpoint_path path/to/model --prompt "" \
    --input_image primary=path/to/image --output_dir ./results

### CLI (multi-view)
python app_cli.py --checkpoint_path path/to/model --prompt "" \
    --input_image primary=path/to/img1 wrist_left=path/to/img2 wrist_right=path/to/img3 \
    --output_dir ./results

### Gradio (auto-adapts to the checkpoint's view configuration)
python app.py --checkpoint_path path/to/model
```

### VLA Training

We provide examples regarding policy training in [`./third-party/lerobot`](./third-party/lerobot).


### Acknowledgements

Thanks to [metaquery](https://github.com/facebookresearch/metaquery), [diffusers](https://github.com/huggingface/diffusers), [lerobot](https://github.com/huggingface/lerobot) for the wonderful open-source codebase.

<!-- ### Citation
If you find ForeAct useful or relevant to your project and research, please kindly cite our paper:
```bibtex

``` -->
