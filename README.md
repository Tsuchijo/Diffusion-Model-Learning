# Diffusion Models

This repo builds Diffusion Models from scratch, explores different types of them (continuous/discrete time, various parameterizations, etc.), implements multiple sampling methods and applies them on 2D toy data and simple image datasets.

## Getting started

After cloning this repo, change directory to the root of the repo and run the following command to install the required packages:

```bash
pip install -e .
```

It will install the `diffusion-models` package. Then you can run explore the scripts in `scripts/` directory. For example, to train a model,
```bash
python scripts/train.py --config <path_to_config>
```
where `<path_to_config>` is the path to a config file. Default config files can be found in `configs/`.