# dots-and-boxes-alpha-zero
Project Deep Reinforcement Learning, Universität Ulm, WiSe 22/23. <br />
**AlphaZero** implementation for the Pen and Paper game **Dots and Boxes**. 

## Installation
```
conda create -n azero_dab python=3.9
conda activate azero_dab

conda install -c anaconda numpy=1.23.4
conda install -c conda-forge tqdm
conda install -c pyyaml

# enable colored print for playing in console
conda install -c conda-forge termcolor=1.1.0
conda install -c anaconda colorama

# pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch  # cpu only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch  # gpu support
```
