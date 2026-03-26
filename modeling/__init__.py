# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from . import bagel_action, qwen2, siglip, autoencoder
from accelerate import init_empty_weights

def load_models(model_path, meta=False):
    llm_config = bagel_action.Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    # 新增action_hidden_size
    llm_config.action_hidden_size = 256

    vit_config = bagel_action.SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vit_config.rope = False
    
    vae_model, vae_config = autoencoder.load_ae(
        local_path=os.path.join(model_path, "ae.safetensors")
    )

    config = bagel_action.BagelConfig(
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
    )

    if meta:
        with init_empty_weights():
            language_model = bagel_action.Qwen2ForCausalLM(llm_config)
            vit_model      = bagel_action.SiglipVisionModel(vit_config)
            model          = bagel_action.Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    else:
        language_model  = bagel_action.Qwen2ForCausalLM(llm_config)
        vit_model       = bagel_action.SiglipVisionModel(vit_config)
        model           = bagel_action.Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    return model, vae_model
