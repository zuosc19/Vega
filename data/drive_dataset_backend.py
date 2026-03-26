
from distributed_iterable_dataset import DistributedIterableDataset
from data_utils import pil_img2rgb
from trajectory import ActionProcessor

from PIL import Image, ImageFile, PngImagePlugin
import torch

import os
import pickle as pkl
import random

# NUM_FRAMES = 14
NUM_HISTORY_FRAMES = 4
NUM_ACTIONS = 8

TEMPLATE_PREFIX = """\
You are an intelligent driving agent. \
You have been given the front-view camera observations for the 3 past steps and the current step at 2 FPS (T=-1.5s, -1.0s, -0.5s, 0.0s). \n\
"""
TEMPLATE_SUFFIX = """\
Your driving instruction is: {instruction} \n\
Predict the appropriate action for the next 8 steps. \
"""

INSTRUCTION_ACTION_TEMPLATE = TEMPLATE_PREFIX + "You have also been given the ego action for the 3 past steps. \n" + TEMPLATE_SUFFIX
INSTRUCTION_TEXT_TEMPLATE = TEMPLATE_PREFIX + """\
Using the coordinate system of the current step, where the X-axis points forward and the Y-axis points left, \
your positions in the 3 past steps (x, y, θ) are: {past_positions}, \
your current velocity (vx, vy) is: {current_velocity}, \
your current acceleration (ax, ay) is: {current_acceleration}. \n\
""" + TEMPLATE_SUFFIX
INSTRUCTION_TEMPLATE = TEMPLATE_PREFIX + "\n" + TEMPLATE_SUFFIX
DEFAULT_INSTRUCTION = """\
Drive smoothly and safely according to the observations. \
"""

# the means and stds above are from the openscene dataset
# MEAN = [2.0142, 0.0026, 0.0024]
# STD = [2.1416, 0.0392, 0.0328]
MEAN = [2.2210, 0.0126, 0.0106]
STD = [1.4761, 0.0655, 0.0563]

class DriveDatasetBackend(DistributedIterableDataset):

    def __init__(
        self, dataset_name, tokenizer, transform, vit_transform, 
        dataset_path, sensor_blobs_path, 
        history_actions="none",
        p_instructions=[("default_instruction", 1.0)], 
        predict_image=True, 
        text_cond_dropout_prob=0.0, 
        vis_cond_dropout_prob=0.0, 
        act_cond_dropout_prob=0.0, 
        local_rank=0, world_size=1, num_workers=8,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.normalizer = ActionProcessor(mean=MEAN, std=STD)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.sensor_blobs_path = sensor_blobs_path
        self.data_paths = self.get_data_paths(dataset_path)

        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vis_cond_dropout_prob = vis_cond_dropout_prob
        self.act_cond_dropout_prob = act_cond_dropout_prob

        if history_actions == "none":
            self._prepare_history = self._prepare_history_no_action
            self.template = INSTRUCTION_TEMPLATE
        elif history_actions == "text":
            self._prepare_history = self._prepare_history_no_action
            self.template = INSTRUCTION_TEXT_TEMPLATE
        elif history_actions == "interleave":
            self._prepare_history = self._prepare_history_interleave_action
            self.template = INSTRUCTION_ACTION_TEMPLATE
        else:   # "separate"
            self._prepare_history = self._prepare_history_separate_action
            self.template = INSTRUCTION_ACTION_TEMPLATE

        self.history_actions = history_actions
        self.p_instructions = p_instructions
        self.predict_image = predict_image

        self.set_epoch()

    def get_data_paths(self, dataset_path):
        data_paths = []
        with open(dataset_path, "rb") as f:
            data = pkl.load(f)
        for key, value in data.items():
            data_paths.append((key, value))
        return data_paths
    
    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'action_list': [],
        }
        return data

    def _skip(self, data, type):
        data['sequence_plan'].append(
            {
                'type': type,
                'skip': True,
            }
        )
        return data

    def _add_text(self, data, text, skip_text=False):
        if skip_text:
            return self._skip(data, "text")
        text_ids = self.tokenizer.encode(text)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'loss': 0,
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, skip_vis=False):
        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            if skip_vis:
                data = self._skip(data, "vae_image")
            else:
                data['sequence_plan'].append(
                    {
                        'type': 'vae_image', 
                        'loss': 0, 
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                )
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            if skip_vis:
                data = self._skip(data, "vit_image")
            else:
                data['sequence_plan'].append(
                    {
                        'type': 'vit_image',
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    },
                )
                vit_image_tensor = self.vit_transform(image)
                height, width = vit_image_tensor.shape[1:]
                data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_action(self, data, action, need_loss=True, need_act=True, skip_act=False):
        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'action', 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )
            data['action_list'].append(action)

        if need_act:
            if skip_act:
                data = self._skip(data, "action")
            else:
                data['sequence_plan'].append(
                    {
                        'type': 'action', 
                        'loss': 0, 
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                )
                data['action_list'].append(action)
            
        return data
    
    def _get_image_path(self, rel_path):
        return os.path.join(self.sensor_blobs_path, rel_path["cam_f0"])

    def _format_text(self, texts):
        instruction_types, weights = zip(*self.p_instructions)
        instruction_type = random.choices(instruction_types, weights=weights, k=1)[0]
        
        instruction = texts.get(instruction_type, DEFAULT_INSTRUCTION)

        return self.template.format(instruction=instruction, **texts)

    # prepare history
    def _prepare_history_no_action(self, data, frame_paths, traj_rel, skip_vis, skip_act):
        for idx in range(NUM_HISTORY_FRAMES):
            image_path = frame_paths[idx]
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(self._get_image_path(image_path))),
                need_loss=False, 
                need_vae=True, 
                need_vit=True, 
                skip_vis=skip_vis,
            )
            
        return data

    def _prepare_history_separate_action(self, data, frame_paths, traj_rel, skip_vis, skip_act):
        for idx in range(NUM_HISTORY_FRAMES):
            image_path = frame_paths[idx]
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(self._get_image_path(image_path))),
                need_loss=False, 
                need_vae=True, 
                need_vit=True, 
                skip_vis=skip_vis,
            )
        action = traj_rel[:NUM_HISTORY_FRAMES-1]
        data = self._add_action(
            data, 
            action, 
            need_loss=False, 
            need_act=True, 
            skip_act=skip_act, 
        )    
        return data

    def _prepare_history_interleave_action(self, data, frame_paths, traj_rel, skip_vis, skip_act):
        for idx in range(NUM_HISTORY_FRAMES):
            image_path = frame_paths[idx]
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(self._get_image_path(image_path))),
                need_loss= False,
                need_vae=True, 
                need_vit=True, 
                skip_vis=skip_vis, 
            )
            if idx != NUM_HISTORY_FRAMES-1:
                action = traj_rel[idx:idx+1]
                data = self._add_action(
                    data, 
                    action, 
                    need_loss=False, 
                    need_act=True, 
                    skip_act=skip_act, 
                )    
        return data

    def parse_data(self, sample_):
        token, sample = sample_
        frame_paths = sample["frame_paths"]
        trajectory = sample["trajectory"]
        texts = sample["texts"]
        traj_rel = self.normalizer.normalize(torch.tensor(trajectory))
        
        data = self._init_data()
        
        skip_text = random.random() < self.text_cond_dropout_prob
        skip_vis = random.random() < self.vis_cond_dropout_prob
        skip_act = random.random() < self.act_cond_dropout_prob

        data = self._prepare_history(data, frame_paths, traj_rel, skip_vis, skip_act)

        # add text
        text = self._format_text(texts)
        data = self._add_text(data, text.rstrip(), skip_text=skip_text)

        # add action
        action = traj_rel[NUM_HISTORY_FRAMES-1:NUM_HISTORY_FRAMES-1+NUM_ACTIONS]     # action is 1 shorter than image
        data = self._add_action(data, action, need_loss=True, need_act=self.predict_image, skip_act=skip_act)

        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(self._get_image_path(frame_paths[NUM_HISTORY_FRAMES-1+NUM_ACTIONS]))),
            need_loss=self.predict_image,
            need_vae=skip_vis and not self.predict_image, 
            need_vit=skip_vis,  # if set to False, when the input vit images are dropped out, the training script may not run correctly
        )
        
        return data

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()

        print(f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}")

        while True:
            for idx, sample in enumerate(file_paths_per_worker):
                try:
                    data = self.parse_data(sample)
                except Exception as e:
                    print(f'Error {e} in rg#{sample[0]}')
                    continue
                data['data_indexes'] = {
                    "data_indexes": idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
