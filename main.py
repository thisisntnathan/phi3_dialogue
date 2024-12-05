#!/usr/bin/env python
# coding: utf-8
# @Filename:  main.py
# @Author:    Nathan Lui
# @Date:      12/04/2024

import os
import warnings

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub.hf_api import HfFolder
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)
from trl import get_kbit_device_map

from utils import (find_target_modules, get_gpu_utilization,
                   get_last_checkpoint, get_max_length,
                   get_number_of_trainable_model_parameters,
                   preprocess_dataset)
from utils import print_main as print

warnings.filterwarnings("ignore", category=UserWarning)


def generate(model, prompt, tokenizer, maxlen=100, sample=True):
    toks = tokenizer(prompt, return_tensors="pt")
    res = model.generate(
        **toks.to(torch.cuda.current_device()),
        max_new_tokens=maxlen,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    ).to("cpu")
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def main():
    # set random seed
    seed = 42

    # load the dataset
    # https://huggingface.co/datasets/knkarthick/dialogsum
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(huggingface_dataset_name)

    # setup dataset with bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # load base model (phi-3.5-mini-instruct)
    model_name = "microsoft/Phi-3.5-mini-instruct"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=get_kbit_device_map(),
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=True,
        attn_implementation="flash_attention_2",
    )

    # create tokenizers
    # https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_bos_token=True, trust_remote_code=True, use_fast=False
    )
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

    # test baseline performance
    index = 42

    prompt = dataset["test"][index]["dialogue"]
    summary = dataset["test"][index]["summary"]

    formatted_prompt = (
        f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"
    )
    res = generate(
        model=original_model,
        prompt=formatted_prompt,
        tokenizer=eval_tokenizer,
    )
    output = res[0].split("\nOutput:\n")[1]

    dash_line = "-".join("" for x in range(100))
    print(dash_line)
    print(f"INPUT PROMPT:\n{formatted_prompt.split('Output')[0]}")
    print(dash_line)
    print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
    print(dash_line)
    print(f"MODEL GENERATION - ZERO SHOT:\n{output}")
    print(dash_line)

    # preprocessing dataset
    print()
    max_length = get_max_length(original_model)

    # use saved datasets if available, else preprocess
    train_path = "./data/train_dataset.hf"
    if os.path.exists(train_path):
        train_dataset = load_from_disk(train_path)
    else:
        train_dataset = preprocess_dataset(
            tokenizer, max_length, seed, dataset["train"]
        )
        train_dataset.save_to_disk(
            train_path
        )  # saves tokenized train dataset as arrow file

    eval_path = "./data/eval_dataset.hf"
    if os.path.exists(eval_path):
        eval_dataset = load_from_disk(eval_path)
    else:
        eval_dataset = preprocess_dataset(
            tokenizer, max_length, seed, dataset["validation"]
        )
        eval_dataset.save_to_disk(
            "./data/eval_dataset.hf"
        )  # saves tokenized eval dataset as arrow file

    print(f"\nShapes of the datasets:")
    print(f"Training: {train_dataset.shape}")
    print(f"Validation: {eval_dataset.shape}")

    print("\nOriginal model parameters:")
    print(get_number_of_trainable_model_parameters(original_model))

    # train the peft adapter
    config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=find_target_modules(original_model),
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    original_model = prepare_model_for_kbit_training(original_model)

    peft_model = get_peft_model(original_model, config)

    print("\nPEFT model parameters:")
    print(get_number_of_trainable_model_parameters(peft_model), "\n")

    output_dir = "./results/checkpoints_1"

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=75,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_strategy="epoch",
        logging_dir="./results/logs_1",
        log_level="info",
        save_strategy="epoch",
        eval_strategy="epoch",
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir="True",
        group_by_length=True,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )

    peft_model.config.use_cache = False

    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    ## lfg
    peft_trainer.train()

    print(get_gpu_utilization())

    # Free memory for merging weights
    del peft_trainer
    torch.cuda.empty_cache()
    print(get_gpu_utilization())


    # qualitiative eval
    ft_model = PeftModel.from_pretrained(
        original_model,
        get_last_checkpoint(output_dir),
        torch_dtype=torch.bfloat16,
        is_trainable=False,
    )

    index = 42
    dialogue = dataset["test"][index]["dialogue"]
    summary = dataset["test"][index]["summary"]

    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    peft_model_res = generate(
        model=ft_model,
        prompt=prompt,
        tokenizer=eval_tokenizer,
    )
    peft_model_output = peft_model_res[0].split("Output:\n")[1]
    prefix, _, _ = peft_model_output.partition("\nEnd.")

    print(dash_line)
    print(f"INPUT PROMPT:\n{prompt.split('Output')[0]}")
    print(dash_line)
    print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
    print(dash_line)
    print(f"PEFT MODEL:\n{prefix}")

    # quantitative eval
    dialogues = dataset["test"][0:10]["dialogue"]
    human_baseline_summaries = dataset["test"][0:10]["summary"]

    original_model_summaries = []
    peft_model_summaries = []

    for dialogue in dialogues:
        prompt = (
            f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
        )

        original_model_res = generate(
            model=original_model,
            prompt=prompt,
            tokenizer=eval_tokenizer,
        )
        original_model_text_output = original_model_res[0].split("Output:\n")[1]

        peft_model_res = generate(
            model=ft_model,
            prompt=prompt,
            tokenizer=eval_tokenizer,
        )
        peft_model_output = peft_model_res[0].split("Output:\n")[1]
        peft_model_text_output, _, _ = peft_model_output.partition("#End")

        original_model_summaries.append(original_model_text_output)
        peft_model_summaries.append(peft_model_text_output)

    # evaluate performance improvement
    rouge = evaluate.load("rouge")

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0 : len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0 : len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print()
    print("ORIGINAL MODEL:")
    print(original_model_results)
    print("PEFT MODEL:")
    print(peft_model_results)

    print()
    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

    improvement = np.array(list(peft_model_results.values())) - np.array(
        list(original_model_results.values())
    )
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f"{key}: {value*100:.2f}%")


if __name__ == "__main__":
    # read HF Hub access token
    with open("hf_token.txt", "r") as f:
        HfFolder.save_token(f.read().strip())

    # set random seed
    set_seed(42)

    # # handle NCCL issues
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # disable Weights and Biases
    os.environ["WANDB_DISABLED"] = "true"

    # use tf32 precision for matmul
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    main()
