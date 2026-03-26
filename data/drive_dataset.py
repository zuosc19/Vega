# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from drive_dataset_backend import DriveDatasetBackend
from transforms import ImageTransform


class DataConfig:
    def __init__(
        self, 
        backend_args, 
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.backend_args = backend_args
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size

class DriveDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,   
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers

        self.use_flex = use_flex
        self.max_num_tokens = max_num_tokens

        for k, v in special_tokens.items():
            setattr(self, k, v)
            
        dataset_args = data_config.backend_args
        if 'image_transform_args' in dataset_args.keys():
            transform = ImageTransform(**dataset_args.pop('image_transform_args'))
            dataset_args['transform'] = transform
        if 'vit_image_transform_args' in dataset_args.keys():
            vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
            dataset_args['vit_transform'] = vit_transform

        self.backend = DriveDatasetBackend(
                dataset_name="drive",
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                **dataset_args, 
            )
        self.dataset_iterator = iter(self.backend)
        self.data_config = data_config
        
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def set_epoch(self, seed):
        self.backend.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
            
            vae_action_tensors          = list(),
            packed_action_position_ids   = list(),
            packed_act_token_indexes    = list(), 
            action_latent_seqlens       = list(), 
            action_timesteps            = list(),
            action_mse_loss_indexes     = list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
            
        # if the model has an action
        if len(sequence_status['vae_action_tensors']) > 0:
            data['vae_action_tensors'] = torch.cat(sequence_status['vae_action_tensors'], dim=0)
            data['packed_action_position_ids'] = torch.cat(sequence_status['packed_action_position_ids'], dim=0)
            data['packed_act_token_indexes'] = torch.tensor(sequence_status['packed_act_token_indexes'])
            data['action_latent_seqlens'] = torch.tensor(sequence_status['action_latent_seqlens'])
            data['action_timesteps'] = torch.tensor(sequence_status['action_timesteps'])
            data['action_mse_loss_indexes'] = torch.tensor(sequence_status['action_mse_loss_indexes'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def __iter__(self):
        while True:
            sample = next(self.dataset_iterator)
            sequence_status = self.pack_sequence(sample)
            data = self.to_tensor(sequence_status)
            data['batch_data_indexes'] = [sample['data_indexes']]
            yield data

    def pack_sequence(self, sample):
        sequence_status = self.set_sequence_status()
        image_tensor_list = sample['image_tensor_list']
        action_list = sample['action_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = 0
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            curr_split_len = 0

            if item.get("skip", False):
                if item["type"] != "text":
                    curr_rope_id += 1
                continue

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                
                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if item['loss'] == 0:
                    curr_rope_id += 1
                    attn_modes.append("full")
                else:
                    attn_modes.append("noise")

            elif item['type'] == 'action':
                action_tensor = action_list.pop(0)

                sequence_status['packed_text_ids'].append(self.start_of_action)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                num_action_tokens = action_tensor.shape[0]
                sequence_status['vae_action_tensors'].append(action_tensor)
                sequence_status['action_latent_seqlens'].append(num_action_tokens)
                sequence_status['packed_action_position_ids'].append(torch.arange(0, num_action_tokens, 1))

                sequence_status['packed_act_token_indexes'].extend(range(curr, curr + num_action_tokens))
                if item['loss'] == 1:
                    sequence_status['action_mse_loss_indexes'].extend(range(curr, curr + num_action_tokens))
                    timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['action_timesteps'].extend([timestep] * num_action_tokens)
                curr += num_action_tokens
                curr_split_len += num_action_tokens

                sequence_status['packed_text_ids'].append(self.end_of_action)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_action_tokens + 2))
                if item['loss'] == 0:
                    curr_rope_id += 1
                    attn_modes.append("full")
                else:
                    attn_modes.append("noise")
                    
            split_lens.append(curr_split_len)
            sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]
            
        if "vae_action_tensors" in data.keys():
            self.vae_action_tensors = data["vae_action_tensors"]
            self.action_timesteps = data["action_timesteps"]
            self.packed_action_position_ids = data["packed_action_position_ids"]
            self.packed_act_token_indexes = data["packed_act_token_indexes"]
            self.action_latent_seqlens = data["action_latent_seqlens"]
            self.action_mse_loss_indexes = data["action_mse_loss_indexes"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()

        if hasattr(self, 'action_timesteps'):
            self.action_mse_loss_indexes = self.action_mse_loss_indexes.pin_memory()
            
        if hasattr(self, 'packed_timesteps') or hasattr(self, 'action_timesteps'):
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()
            
        if hasattr(self, 'vae_action_tensors'):
            self.vae_action_tensors = self.vae_action_tensors.pin_memory()
            self.action_timesteps = self.action_timesteps.pin_memory()
            self.packed_action_position_ids = self.packed_action_position_ids.pin_memory()
            self.packed_act_token_indexes = self.packed_act_token_indexes.pin_memory()
            self.action_latent_seqlens = self.action_latent_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)

        if hasattr(self, 'action_timesteps'):
            self.action_mse_loss_indexes = self.action_mse_loss_indexes.to(device)
            
        if hasattr(self, 'packed_timesteps') or hasattr(self, 'action_timesteps'):
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)
            
        if hasattr(self, 'vae_action_tensors'):
            self.vae_action_tensors = self.vae_action_tensors.to(device)
            self.action_timesteps = self.action_timesteps.to(device)
            self.packed_action_position_ids = self.packed_action_position_ids.to(device)
            self.packed_act_token_indexes = self.packed_act_token_indexes.to(device)
            self.action_latent_seqlens = self.action_latent_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
            
        if hasattr(self, 'vae_action_tensors'):
            data['vae_action_tensors'] = self.vae_action_tensors
            data['action_timesteps'] = self.action_timesteps
            data['packed_action_position_ids'] = self.packed_action_position_ids
            data['packed_act_token_indexes'] = self.packed_act_token_indexes
            data['action_latent_seqlens'] = self.action_latent_seqlens
            data['action_mse_loss_indexes'] = self.action_mse_loss_indexes

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
