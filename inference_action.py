# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
import json
from collections import defaultdict
# import shutil

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

import sys
sys.path.append("/path/to/Vega")
sys.path.append("/path/to/Vega/data")

from data.drive_dataset_eval import DriveDatasetForEvalIterable, eval_collate_wrapper
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel_action import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling import load_models
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
)

from inferencer import InterleaveInferencerAction

model_path = "/path/to/Bagel-7B-MoT"
ckpt_path = "/path/to/checkpoints/ema.safetensors"

resume_from_ema = True

dataset_path = "/path/to/navtest_instruct.pkl"
sensor_blobs_path = "/path/to/nuplan-v1.1/sensor_blobs"

best_of_k = 1
history_actions = "none"
instruction_type = "rule_based_instruction"
first_n = None

output_path = "./results"

vae_transform = ImageTransform(256, 128, 16)
vit_transform = ImageTransform(224, 112, 14)

action_kwargs=dict(
    history_actions=(history_actions not in ["none", "text"]), 
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_act_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=20,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    enable_taylorseer=False,
)

global_seed = 0
sharding_strategy = "HYBRID_SHARD" # HYBRID_SHARD or NO_SHARD if num_shard == 1
backward_prefetch = "BACKWARD_PRE" # BACKWARD_PRE or NO_PREFETCH
prefetch_factor = 2
cpu_offload = False

@dataclass
class EvalArguments:
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )

def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser(EvalArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    if dist.get_rank() == 0:
        os.makedirs(output_path, exist_ok=True)
        logger = create_logger(output_path, dist.get_rank())
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Eval arguments {eval_args}')
    set_seed(global_seed)

    model, vae_model = load_models(model_path, meta=False)
    logger.info(f"Model: \n{model}")
    
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    fsdp_config = FSDPConfig(
        sharding_strategy=sharding_strategy,
        backward_prefetch=backward_prefetch,
        cpu_offload=cpu_offload,
        num_replicate=eval_args.num_replicate,
        num_shard=eval_args.num_shard,
    )
    model, _ = FSDPCheckpoint.try_load_ckpt(
        ckpt_path, logger, model, resume_from_ema=resume_from_ema,
    )
    fsdp_model = fsdp_wrapper(model, fsdp_config)

    if dist.get_rank() == 0:
        print(fsdp_model)

    dataset = DriveDatasetForEvalIterable(
        dataset_path,
        sensor_blobs_path,
        max_epochs=best_of_k,
        history_actions=history_actions, 
        instruction_type=instruction_type, 
        first_n=first_n, 
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=eval_args.num_workers,
    )
    dataset.set_epoch(seed=global_seed)
    loader = DataLoader(
        dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=eval_args.num_workers,
        pin_memory=True,
        collate_fn=eval_collate_wrapper(),
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )
    logger.info(f"Loaded dataset with size {len(dataset)}")

    vae_model.to(device).eval()
    fsdp_model.eval()

    # inferencer
    inferencer = InterleaveInferencerAction(model=fsdp_model, 
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        device=device,
    )

    predictions = defaultdict(list)

    start_time = time()
    logger.info(f"Starting eval ...")
    with torch.no_grad():
        for curr_step, data in enumerate(loader):
            action_pred_ = inferencer.inference_action(images=data["images"],
                instruction=data["text"],
                past_actions=data.pop("past_actions", None),
                **action_kwargs
            )
            action_pred = dataset.normalizer.denormalize(action_pred_.cpu()).tolist()
            action = dataset.normalizer.denormalize(data["action"].cpu()).tolist()

            predictions[data["token"]].append([action_pred, action, data["text"]])

            if (curr_step+1) % eval_args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                sec_per_step = (end_time - start_time) / eval_args.log_every
                logger.info(f"[Step={(curr_step+1):06d}] {sec_per_step:.2f} seconds per step")
                start_time = time()

    output_file = os.path.join(output_path, f"rank_{dist.get_rank():04d}.json")
    predictions.pop(dataset.PAD_TOKEN, None)
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
            
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
