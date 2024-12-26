# Mask-Pack

## Environment
- OS: Ubuntu 20.04.1
- Python: 3.9.15
- PyTorch: 2.4.1+cu118
- stable-baselines3: 2.3.2
> Note: the training results might be defferent with different hearware environment and OS, under the same .yaml setting file.

## Installation
```bash
# only cpu
bash scripts/install.sh -c 0

# use cuda (version: 11.8)
bash scripts/install.sh -c 11.8
```

## Usage example
```bash
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
```

## Using docker images
### Build docker image
Build CPU image:
```bash
make docker-cpu
```
Build GPU image (with nvidia-docker):
```bash
make docker-gpu
```

### Run the images (CPU/GPU)
Run the nvidia-container-toolkit GPU image
```bash
docker run -it --rm --gpus=all --network host --ipc=host --mount src=$(pwd),target=/home/user/maskpack,type=bind jeepway/maskpack-gpu:latest bash -c "cd /home/user/maskpack && ls && pwd && /bin/bash"
```
Or, use make command to run with the shell file
```bash
make docker-run-gpu
```
Run the docker CPU image
```bash
docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/home/user/maskpack,type=bind jeepway/maskpack-cpu:latest bash -c "cd /home/user/maskpack && ls && pwd && /bin/bash"
```
Or, use make command to run with the shell file
```bash
make docker-run-cpu
```
After running the above command, you will enter the terminal of the docker image.

### Run .yaml setting file
Run the .yaml setting in the [setting folder](https://github.com/JeepWay/mask-pack/tree/main/settings), for example:
```bash
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
```
After running the above command, you will be asked to choose the visualization mode like below.
```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```
If you want to visualize the training results on the wandb website, you can choose the second option, and then you will be asked to paste the API key of your wandb account. 

If you don't have one, you can choose the first option to create a new account.

If you just want to save the training results locally, you can choose the third option.

## Compared algorithms
you can watch the source code of the compared algorithms in this [repository](https://github.com/JeepWay/mask-pack-compare-algorithm)
1. Zhao-2D (transform from 3D to 2D of the original paper)
   * Paper: [Online 3D Bin Packing with Constrained Deep Reinforcement Learning](https://arxiv.org/abs/2006.14978)
   * GitHub: [Online-3D-BPP-DRL](https://github.com/alexfrom0815/Online-3D-BPP-DRL)
2. Deep-Pack
   * Paper: [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)
   * GitHub: [Deep-Pack (unofficial implementation)](https://github.com/JeepWay/DeepPack)
3. Deep-Pack-with-mask
   * Based on Deep-Pack, we add the action mask to the original algorithm to reduce the selection of invalid actions, for the purpose of fair comparison with Mask-Pack.

## Reference
