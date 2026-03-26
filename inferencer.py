# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel_action.qwen2_navit_action import NaiveCache

def dict_to(dict_, device=None, float_to_bfloat16=True):
    for key, value in dict_.items():
        if isinstance(value, torch.Tensor):
            if device is not None:
                value = value.to(device)
            if value.dtype == torch.float and float_to_bfloat16:
                value = value.to(torch.bfloat16)
            dict_[key] = value
    return dict_


class InterleaveInferencerAction:
    def __init__(self, model, vae_model, tokenizer, 
        vae_transform, vit_transform, new_token_ids,
        device="cuda"
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        self.device = device

        # normalize_method = "standard"
        # self.normalizer = ActionProcessor(**NORMALIZE_PARAMS[normalize_method])
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        generation_input = dict_to(generation_input, self.device)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_action(self, action, gen_context):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_action(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            actions=[action],
            new_token_ids=self.new_token_ids,
        )
        generation_input = dict_to(generation_input, self.device)
        past_key_values = self.model.forward_cache_update_act(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context
    
    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_act_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_act_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        
        # act cfg
        cfg_act_past_key_values = cfg_act_precontext['past_key_values']
        kv_lens_cfg = cfg_act_precontext['kv_lens']
        ropes_cfg = cfg_act_precontext['ropes']
        generation_input_cfg_act = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            cfg_act_past_key_values=cfg_act_past_key_values, 
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_act_scale=cfg_act_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            cfg_act_packed_position_ids=generation_input_cfg_act['cfg_packed_position_ids'],
            cfg_act_packed_query_indexes=generation_input_cfg_act['cfg_packed_query_indexes'],
            cfg_act_key_values_lens=generation_input_cfg_act['cfg_key_values_lens'],
            cfg_act_packed_key_value_indexes=generation_input_cfg_act['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    @torch.no_grad()
    def gen_action(
        self, 
        num_action_tokens, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_act_scale=1.5,
        use_cfg_act=False,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_act_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_act_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            action_lengths=[num_action_tokens], 
            new_token_ids=self.new_token_ids,
        ) 
        generation_input = dict_to(generation_input, self.device)
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_act_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            action_lengths=[num_action_tokens], 
        )
        generation_input_cfg_text = dict_to(generation_input_cfg_text, self.device)

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_act_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            action_lengths=[num_action_tokens], 
        )
        generation_input_cfg_img = dict_to(generation_input_cfg_img, self.device)
        
        # act cfg
        if use_cfg_act:
            cfg_act_past_key_values = cfg_act_precontext['past_key_values']
            kv_lens_cfg = cfg_act_precontext['kv_lens']
            ropes_cfg = cfg_act_precontext['ropes']
            generation_input_cfg_act = self.model.prepare_act_latent_cfg(
                curr_kvlens=kv_lens_cfg,
                curr_rope=ropes_cfg, 
                action_lengths=[num_action_tokens], 
            )
            generation_input_cfg_act = dict_to(generation_input_cfg_act, self.device)
            cfg_act_input = dict(
                cfg_act_past_key_values=cfg_act_past_key_values,
                cfg_act_packed_position_ids=generation_input_cfg_act['cfg_packed_position_ids'],
                cfg_act_packed_query_indexes=generation_input_cfg_act['cfg_packed_query_indexes'],
                cfg_act_key_values_lens=generation_input_cfg_act['cfg_key_values_lens'],
                cfg_act_packed_key_value_indexes=generation_input_cfg_act['cfg_packed_key_value_indexes'],
            )
        else:
            cfg_act_input = dict()

        unpacked_latent = self.model.generate_action(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_act_scale=cfg_act_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            use_cfg_act=use_cfg_act, 
            **cfg_act_input, 
            enable_taylorseer=enable_taylorseer,
        )

        action = unpacked_latent[0]
        return action
        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)

        device = self.vae_model.encoder.conv_in.weight.device
        image = self.vae_model.decode(latent.to(device))
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def inference_image(
        self,
        images: List[Image.Image],
        instruction: str,
        action: torch.Tensor, 

        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_act_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_action_tokens=8,
        enable_taylorseer=False,
    ):

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        cfg_act_context = deepcopy(gen_context)

        prompt = instruction

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for image in images:
                input_term = self.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = self.update_context_image(input_term, gen_context)
                image_shapes = input_term.size[::-1]

            cfg_text_context = deepcopy(gen_context)
            gen_context = self.update_context_text(prompt, gen_context)
            cfg_img_context = self.update_context_text(prompt, cfg_img_context)

            cfg_act_context = deepcopy(gen_context)
            # gen_context = self.update_context_action(self.normalizer.normalize(action), gen_context)
            gen_context = self.update_context_action(action, gen_context)
            cfg_img_context = self.update_context_action(action, cfg_img_context)
            cfg_text_context = self.update_context_action(action, cfg_text_context)
            
            img = self.gen_image(
                image_shapes, 
                gen_context, 
                cfg_text_precontext=cfg_text_context, 
                cfg_img_precontext=cfg_img_context,
                cfg_act_precontext=cfg_act_context,

                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_act_scale=cfg_act_scale,
                cfg_interval=cfg_interval, 
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
            )
            return img

    @torch.no_grad()
    def inference_action(
        self,
        images: List[Image.Image],
        instruction: str,
        past_actions: Optional[List[torch.Tensor]] = None,

        history_actions=False,
        interleave=False, 
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_act_scale=1.5, 
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_action_tokens=8,
        enable_taylorseer=False,
    ):
        use_cfg_act = (past_actions is not None and history_actions)

        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        if use_cfg_act:
            cfg_act_context = deepcopy(gen_context)
        else:
            cfg_act_context = None

        prompt = instruction

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if use_cfg_act:
                assert len(past_actions) == len(images) - 1, f"{len(images)} images, {len(past_actions)} past actions"

                if interleave:
                    for idx in range(len(images)):
                        image_ = self.vae_transform.resize_transform(pil_img2rgb(images[idx]))
                        gen_context = self.update_context_image(image_, gen_context)
                        cfg_text_context = self.update_context_image(image_, cfg_text_context)
                        cfg_act_context = self.update_context_image(image_, gen_context)
                        if idx != len(images) - 1:
                            past_action = past_actions[idx:idx+1]
                            gen_context = self.update_context_action(past_action, gen_context)
                            cfg_text_context = self.update_context_action(past_action, cfg_text_context)
                            cfg_img_context = self.update_context_action(past_action, cfg_img_context)
                else:
                    for idx in range(len(images)):
                        image_ = self.vae_transform.resize_transform(pil_img2rgb(images[idx]))
                        gen_context = self.update_context_image(image_, gen_context)
                        cfg_text_context = self.update_context_image(image_, cfg_text_context)
                        cfg_act_context = self.update_context_image(image_, gen_context)
                    gen_context = self.update_context_action(past_actions, gen_context)
                    cfg_text_context = self.update_context_action(past_actions, cfg_text_context)
                    cfg_img_context = self.update_context_action(past_actions, cfg_img_context)
            else:
                for idx in range(len(images)):
                    image_ = self.vae_transform.resize_transform(pil_img2rgb(images[idx]))
                    gen_context = self.update_context_image(image_, gen_context)
                    cfg_text_context = self.update_context_image(image_, cfg_text_context)


            if prompt is not None:
                gen_context = self.update_context_text(prompt, gen_context)
                cfg_img_context = self.update_context_text(prompt, cfg_img_context)
                if use_cfg_act:
                    cfg_act_context = self.update_context_text(prompt, cfg_act_context)

            action = self.gen_action(
                num_action_tokens, 
                gen_context, 
                cfg_text_precontext=cfg_text_context, 
                cfg_img_precontext=cfg_img_context,
                cfg_act_precontext=cfg_act_context,
                use_cfg_act=use_cfg_act,

                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_act_scale=cfg_act_scale, 
                cfg_interval=cfg_interval, 
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
            )
            return action
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
