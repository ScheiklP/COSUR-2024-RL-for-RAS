# COSUR 2024 - Reinforcement Learning for Robot-Assisted Surgery
This repository contains the code for the practical lab part of the Reinforcement Learning for Robot-Assisted Surgery part of the
[cosur: summer school on control of surgical robots](https://metropolis.scienze.univr.it/cosur-2024/) -
dvrk from zero to hero held 15-18 July 2024 – Verona, Strada Le Grazie 15, ITALY.

## Getting Started
1. Install Conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3*.sh
conda init
```

2. Create a Conda environment
```bash
conda create -n cosur_rl python=3.10
conda activate cosur_rl
```

3. Install the dependencies of this repo
```bash
pip install -r requirements.txt
```

4. Test the installation
```bash
python3 cosur/dvrk_point_reach_env.py
```

## Acknowledgements
- PSM and ECM mesh files and URDF files from [https://github.com/WPI-AIM/dvrk_env](https://github.com/WPI-AIM/dvrk_env)
- The cloth mesh files and inspiration for grasping deformable objects from [https://github.com/DanielTakeshi/deformable-ravens](https://github.com/DanielTakeshi/deformable-ravens)
