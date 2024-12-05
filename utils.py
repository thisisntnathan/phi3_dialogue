#!/usr/bin/env python
# coding: utf-8
# @Filename:  utils.py
# @Author:    Nathan Lui
# @Date:      12/05/2024

import os
import re

from functools import partial

import torch
from datasets import Dataset
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_gpu_utilization() -> str:
    """Get the GPU memory utilization
    
    :return str msg: Message containing the GPU memory utilization
    :rtype: str
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def print_main(s: str = "") -> None:
    """Check if this process is running on GPU:0, if so print the message
    
    :param str s: Message to print
    :return: None
    """
    if torch.cuda.current_device() == 0:
        print(s)


def get_number_of_trainable_model_parameters(model) -> str:
    """Get the number of trainable model parameters

    :param model: Model to get the number of trainable parameters from
    :return str msg: Message containing the number of trainable model parameters
    :rtype: str
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")


def find_target_modules(model) -> list[str]:
    """Find all the modules in the model that are of type 'Linear4bit'

    :param model: Model to search for target modules
    :return list modules: List of target module names
    :rtype: list[str]
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


def create_prompt_formats(sample) -> dict:
    """Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters

    :param dict sample: prompt to format
    :return dict sample: formatted prompt
    :rtype: dict
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
def get_max_length(model) -> int:
    """Get the maximum length of the model
    
    :param model: Model to get the maximum length from
    :return int max_length: Maximum length of the model
    :rtype: int
    """
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
    """Tokenize a batch
    
    :param batch: Batch to tokenize
    :param tokenizer: Model Tokenizer
    :param max_length: Maximum number of tokens to emit from tokenizer
    :return btch: Tokenized batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer, data, max_length=100, seed=42) -> Dataset:
    """Format & tokenize it so it is ready for training

    :param AutoTokenizer tokenizer: Model Tokenizer
    :param Dataset data: Dataset to preprocess
    :param int max_length: Maximum number of tokens to emit from tokenizer
    :param int seed: Seed for shuffling the dataset
    :return Dataset dataset: Preprocessed dataset
    :rtype: Dataset
    """

    # Add prompt to each sample
    print_main("Preprocessing dataset...")
    dataset = data.map(create_prompt_formats)  # , batched=True)

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


def get_last_checkpoint(directory) -> os.PathLike:
    """Get the latest checkpoint in the directory

    :param directory: Directory to search for checkpoints
    :return: Path to the latest checkpoint or None if no checkpoint is found
    :rtype: Union[str, None]
    """

    # regular expression to match "checkpoint-<number>"
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