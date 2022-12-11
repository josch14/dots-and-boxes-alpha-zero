# dots-and-boxes-alpha-zero
Project Deep Reinforcement Learning, Universität Ulm, WiSe 22/23. <br />
**AlphaZero** implementation for the Pen and Paper game **Dots and Boxes**. 

## Installation
```
conda create -n azero_dab python=3.9
conda activate azero_dab

conda install -c anaconda numpy=1.23.4
conda install -c conda-forge tqdm
conda install -c anaconda yaml

# enable colored print for playing in console
conda install -c conda-forge termcolor=1.1.0
conda install -c anaconda colorama

# pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```