

The [`analysis`](analysis/) folder contains the code, configuration files, and instructions required to reproduce the paper’s results.

The [`demo`](demo/) folder contains a lightweight CPU-only run that should complete in approximately 1-2 minutes.

Installing the software environment should take only a few minutes, assuming that a Conda distribution is already installed.


## Software Installation

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or another
Conda distribution, then clone this repository and create the supplied
environment:

`conda env create --code/environment.yml`

`conda activate populations_env`

for macOS, after the installation and activation of the environment  
run: 

`python -m pip install --no-deps gwpopulation-pipe==0.4.3`

`conda install --override-channels --channel conda-forge "python-htcondor=24.12.4"`

## System requirements

The full population analysis was performed and tested on:

- Rocky Linux 8.10 (Green Obsidian)
- Python version specified in `code/environment.yml`
- HTCondor computing cluster

The analysis has not been tested on other operating systems. The data-download
and post-processing scripts are expected to work on standard Linux and macOS
systems.


## Hardware used for the analysis

The full population inference was performed using:

- One NVIDIA A100-SXM4-80GB GPU with 80 GB of GPU memory
- The JAX GPU backend
- An HTCondor computing cluster running Rocky Linux 8.10

The post-processing scripts do not require a GPU.


