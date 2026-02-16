# ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

We build the training code for **ForeAct** VLA models on top of [LeRobot](https://github.com/huggingface/lerobot).

## Installation

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

cd third-party/lerobot
pip install -e .
```

## Dataset

Our pre-processed real-world training data can be accessed on huggingface:

```bash
hf download mit-han-lab/ForeAct_VLA_Dataset      --repo-type=dataset  
hf download mit-han-lab/ForeAct_VLA_Dataset_flat --repo-type=dataset
```

## Training

Example training scripts for $\pi_{0.5}$ models are provided in [`./scripts`](./scripts):

- `run_flat_100k.sh` — baseline policy
- `run_sub_task_100k.sh` — policy trained with sub-task descriptions
- `run_goal_100k.sh` — policy trained with visual foresight goals

