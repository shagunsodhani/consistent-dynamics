# Consistent Dynamics

Code for the paper "[Learning Powerful Policies by Using Consistent Dynamics Model](http://spirl.info/2019/camera-ready/spirl_camera-ready_15.pdf)" presented at Workshop on "Structure & Priors in Reinforcement Learning" at ICLR 2019. 

Arxiv link coming soon!

## Code Setup

* Copy `config/sample.config.yaml` to `config/config.yaml`. For testing, no changes are needed in the newly created `config.yaml`.
* Install requirements using `pip3 install -r requirements.txt`.
* Following packages needs to be installed from source:
    * https://github.com/openai/baselines#installation
* From the root dir, run `PYTHONPATH=$PWD python3 codes/app/main.py`

## Notes

* All the logic has a single entry point `codes/app/main.py`. What task is to be performed is controlled via the config files.

## Setup Issues

* If you get an error related to `patchelf`, try `conda install -c anaconda patchelf`
