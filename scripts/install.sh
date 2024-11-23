#!/bin/bash
#bash install.sh -c 0
#bash install.sh -c 11.8

. ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate base
echo Y | conda create -n MaskPack python=3.9.15
conda activate MaskPack

supported_cuda_versions=(false, 0, 11.8)
while getopts ":c:" opt; do
    case $opt in
    c)
        if [[ "${supported_cuda_versions[@]}" =~ "${OPTARG}" ]] ; then
            cuda="${OPTARG}"
        else
            echo "Invalid CUDA version: ${OPTARG}. Supported versions are: ${supported_cuda_versions[@]}"
            exit 1
        fi
        ;;
    :)
        echo "Option -$OPTARG requires an argument, e.g., -c 11.8."
        exit 1
        ;;
    \?)
        echo "Invalid option: -$OPTARG."
        exit 1
        ;;
    esac
done

if [[ "${cuda}" == "false" || "${cuda}" == "0" ]] ; then
    echo "Install PyTorch with cpu version."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Install PyTorch with cuda ${cuda} version."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

pip install stable-baselines3==2.3.2
pip install tensorboard==2.18.0
pip install rl_zoo3==2.3.0
pip install pyyaml
pip install scipy==1.13.0
pip install wandb==0.18.7
