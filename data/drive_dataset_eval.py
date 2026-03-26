
from distributed_iterable_dataset import DistributedIterableDataset
from data_utils import pil_img2rgb
from drive_dataset_backend import (ActionProcessor, 
    INSTRUCTION_ACTION_TEMPLATE,
    INSTRUCTION_TEXT_TEMPLATE, 
    INSTRUCTION_TEMPLATE, 
    DEFAULT_INSTRUCTION, 
    NUM_HISTORY_FRAMES, 
    NUM_ACTIONS, 
    MEAN, STD)
    
import numpy as np
from PIL import Image, ImageFile, PngImagePlugin
import torch

import os
import pickle as pkl
import random

# For single sample inference
class DriveDatasetForEval:

    def __init__(self, dataset_path, sensor_blobs_path, first_n=None, history_actions="none", instruction_type="default_instruction"):
        self.dataset_path = dataset_path
        self.sensor_blobs_path = sensor_blobs_path
        self.history_actions = history_actions
        self.instruction_type = instruction_type
        self.template = INSTRUCTION_TEMPLATE if history_actions == "none" else INSTRUCTION_TEXT_TEMPLATE if history_actions == "text" \
            else INSTRUCTION_ACTION_TEMPLATE

        self.normalizer = ActionProcessor(mean=MEAN, std=STD)
        self._load_data(first_n)

        
    def _load_data(self, first_n):
        dataset_path = self.dataset_path
        with open(dataset_path, "rb") as f:
            self.data = pkl.load(f)
        self.tokens = list(self.data.keys())
        
        if first_n is not None:
            self.tokens = self.tokens[:first_n]

    def __len__(self):
        return len(self.tokens)
    
    def _get_image_path(self, rel_path):
        return os.path.join(self.sensor_blobs_path, rel_path["cam_f0"])
    
    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, instruction=None):
        if isinstance(idx, int):
            token = self.tokens[idx]
        else:
            token = idx
        sample = self.data[token]

        frame_paths = sample["frame_paths"]
        trajectory = sample["trajectory"]
        traj_rel = self.normalizer.normalize(torch.tensor(trajectory))
        
        images = [Image.open(self._get_image_path(image_path)) for image_path in frame_paths]
        
        past_actions = traj_rel[:NUM_HISTORY_FRAMES-1]

        FUTURE_FRAME_IDX = NUM_ACTIONS
        future_image = Image.open(self._get_image_path(frame_paths[NUM_HISTORY_FRAMES-1+FUTURE_FRAME_IDX]))

        texts = sample["texts"]
        if instruction is None:
            instruction = texts.get(self.instruction_type, DEFAULT_INSTRUCTION)
        text = self.template.format(instruction=instruction, **texts)

        action = traj_rel[NUM_HISTORY_FRAMES-1:NUM_HISTORY_FRAMES-1+NUM_ACTIONS]

        return {
            "token": token,
            "images": images,
            "future_image": future_image,
            "text": text, 
            "past_actions": past_actions,
            "action": action,
        }

# For benchmarking
class DriveDatasetForEvalIterable(DistributedIterableDataset):
    DATASET_NAME = "drive_dataset_for_eval"
    PAD_TOKEN = "<|PAD|>"

    def __init__(self, dataset_path, sensor_blobs_path,
        max_epochs=1, 
        history_actions="none",
        instruction_type="default_instruction", 
        first_n=None, 
        local_rank=0, world_size=1, num_workers=8
    ):
        super().__init__(self.DATASET_NAME, local_rank, world_size, num_workers)
        self.dataset_path = dataset_path
        self.sensor_blobs_path = sensor_blobs_path
        self.max_epochs = max_epochs
        self.normalizer = ActionProcessor(mean=MEAN, std=STD)
        self._load_data(first_n)

        self.history_actions = history_actions
        self.instruction_type = instruction_type
        self.template = INSTRUCTION_TEMPLATE if history_actions == "none" else INSTRUCTION_TEXT_TEMPLATE if history_actions == "text" \
            else INSTRUCTION_ACTION_TEMPLATE
        
    def _load_data(self, first_n):
        dataset_path = self.dataset_path
        with open(dataset_path, "rb") as f:
            self.data = pkl.load(f)
        self.tokens = list(self.data.keys())
        
        if first_n is not None:
            self.tokens = self.tokens[:first_n]

        """
        extending the dataset to match the number of GPUs, or the last samples will be dropped
        """
        pad_unit = self.world_size * self.num_workers
        pad_length = (pad_unit - len(self.tokens) % pad_unit) % pad_unit
        self.data_paths = self.tokens + [self.PAD_TOKEN] * pad_length

    def __len__(self):
        return len(self.data_paths)
    
    def _get_image_path(self, rel_path):
        return os.path.join(self.sensor_blobs_path, rel_path["cam_f0"])

    def _get_sample(self, idx_or_token):
        if isinstance(idx_or_token, int):
            token = self.data_paths[idx_or_token]
        else:
            token = idx_or_token

        if token == self.PAD_TOKEN:
            token_ = random.choice(self.tokens)
        else:
            token_ = token
        
        return token, self.data[token_]

    def _format_text(self, texts):
        instruction = texts.get(self.instruction_type, DEFAULT_INSTRUCTION)
        return self.template.format(instruction=instruction, **texts)

    def __getitem__(self, idx_or_token):
        token, sample = self._get_sample(idx_or_token)

        frame_paths = sample["frame_paths"]
        trajectory = sample["trajectory"]
        texts = sample["texts"]
        traj_rel = self.normalizer.normalize(torch.tensor(trajectory))
        
        history_image_paths = frame_paths[:NUM_HISTORY_FRAMES]
        images = [Image.open(self._get_image_path(image_path)) for image_path in history_image_paths]

        past_actions = traj_rel[:NUM_HISTORY_FRAMES-1]

        # FUTURE_FRAME_IDX = 1
        # future_image = Image.open(self._get_image_path(frame_paths[NUM_HISTORY_FRAMES-1+FUTURE_FRAME_IDX]))

        text = self._format_text(texts)

        NUM_ACTIONS = 8
        action = traj_rel[NUM_HISTORY_FRAMES-1:NUM_HISTORY_FRAMES-1+NUM_ACTIONS]

        return {
            "token": token,
            "images": images,
            # "future_image": future_image,
            "text": text, 
            "past_actions": past_actions,
            "action": action,
        }

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"Padding from {len(self.tokens)} to {len(self.data_paths)}"
        )

        for epoch in range(self.max_epochs):
            for global_row_group_idx, token in enumerate(file_paths_per_worker):
                try:
                    data = self[token]
                except Exception as e:
                    print(f'Error {e} in rg#{token}')
                    continue
                data['data_indexes'] = {
                    "data_indexes": global_row_group_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

def eval_collate_wrapper():
    def collate_fn(batch):
        return batch[0]
    return collate_fn
