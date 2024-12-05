#!/usr/bin/env python
# coding: utf-8
# @Filename:  utils.py
# @Author:    Nathan Lui
# @Date:      12/04/2024

import os
import re

from functools import partial

import torch
from transformers import AutoTokenizer
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def print_main(s: str = ""):
    """Check if this process is running on GPU:0, if so print the message"""
    if torch.cuda.current_device() == 0:
        print(s)


def get_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")


def find_target_modules(model):
    """Find all the modules in the model that are of type 'Linear4bit'
    :param model: Model to search for target modules
    """
    # initialize a set to store unique layers
    unique_layers = set()

    # iterate over all named modules in the model    
    for name, module in model.named_modules():
        # check if the module type contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # add the layer type to the set of unique layers            
            unique_layers.add(layer_type)

    # return the set of unique layers converted to a list
    return list(unique_layers)


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. \
        Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['dialogue']}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample['summary']}"
    end = f"{END_KEY}"

    parts = [
        part for part in [blurb, instruction, input_context, response, end] if part
    ]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print_main(f"Found max lenth: {max_length}")
            return max_length
    if not max_length:
        max_length = 1024
        print_main(f"Using default max length: {max_length}")
        return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """Tokenizing a batch"""
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print_main("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["id", "topic", "dialogue", "summary"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def get_last_checkpoint(directory):
    """Get the latest checkpoint in the directory
    :param directory: Directory to search for checkpoints
    """

    # Regular expression to match "checkpoint-<number>"
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    
    latest_checkpoint = None
    highest_step = -1

    for filename in os.listdir(directory):
        match = checkpoint_pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step > highest_step:
                highest_step = step
                latest_checkpoint = filename

    if latest_checkpoint:
        return os.path.join(directory, latest_checkpoint)
    else:
        return None