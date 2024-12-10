# Phi3_Dialogue

A small pet project fine-tuning Microsoft's Phi3.5-mini-instruct to summarize dialogue. Developed as part of a set of a benchmarking workflows I wrote for the C3 compute hardware upgrade.  

## Environment

Use the provided `environment.yml` file to set up the running environment using Conda.  

```bash
conda env create -f environment.yml
conda activate phi3_dialogue
```

## Usage

This script can run on multiple GPUs using the 🤗 Accelerate library. To launch distributed training, make sure to set up 🤗 Accelerate with `accelerate config` before calling main.py the first time.  

After configuring multi-GPU/node acceleration launch the script from the command line with:

```bash
accelerate launch --num_processes <number of gpus> main.py {--args}
```
