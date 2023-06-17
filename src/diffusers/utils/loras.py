import torch
from safetensors.torch import load_file
from collections import defaultdict
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

def convert_name_to_bin(name):
    
    # down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up
    new_name = name.replace(LORA_PREFIX_UNET+'_', '')
    new_name = new_name.replace('.weight', '')
    
    # ['down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q', 'lora.up']
    parts = new_name.split('.')
    
    #parts[0] = parts[0].replace('_0', '')
    if 'out' in parts[0]:
        parts[0] = "_".join(parts[0].split('_')[:-1])
    parts[1] = parts[1].replace('_', '.')
    
    # ['down', 'blocks', '0', 'attentions', '0', 'transformer', 'blocks', '0', 'attn1', 'to', 'q']
    # ['mid', 'block', 'attentions', '0', 'transformer', 'blocks', '0', 'attn2', 'to', 'out']
    sub_parts = parts[0].split('_')

    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_
    new_sub_parts = ""
    for i in range(len(sub_parts)):
        if sub_parts[i] in ['block', 'blocks', 'attentions'] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
            if 'attn' in sub_parts[i]:
                new_sub_parts += sub_parts[i] + ".processor."
            else:
                new_sub_parts += sub_parts[i] + "."
        else:
            new_sub_parts += sub_parts[i] + "_"
    
    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.up
    new_sub_parts += parts[1]
    
    new_name =  new_sub_parts + '.weight'
    
    return new_name


def safetensors_to_bin(safetensor_path, bin_path):
    
    bin_state_dict = {}
    safetensors_state_dict = load_file(safetensor_path)
        
    for key_safetensors in safetensors_state_dict:
        # these if are required  by current diffusers' API
        # remove these may have negative effect as not all LoRAs are used
        if 'text' in key_safetensors:
            continue
        if 'unet' not in key_safetensors:
            continue
        if 'transformer_blocks' not in key_safetensors:
            continue
        if 'ff_net' in key_safetensors or 'alpha' in key_safetensors:
            continue
        key_bin = convert_name_to_bin(key_safetensors)
        bin_state_dict[key_bin] = safetensors_state_dict[key_safetensors]
    
    torch.save(bin_state_dict, bin_path)

    
def convert_name_to_safetensors(name):
    
    # ['down_blocks', '0', 'attentions', '0', 'transformer_blocks', '0', 'attn1', 'processor', 'to_q_lora', 'up', 'weight']
    parts = name.split('.')
    
    # ['down_blocks', '_0', 'attentions', '_0', 'transformer_blocks', '_0', 'attn1', 'processor', 'to_q_lora', 'up', 'weight']
    for i in range(len(parts)):
        if parts[i].isdigit():
            parts[i] = '_' + parts[i]
        if "to" in parts[i] and "lora" in parts[i]:
            parts[i] = parts[i].replace('_lora', '.lora')
        
    new_parts = []
    for i in range(len(parts)):
        if i == 0:
            new_parts.append(LORA_PREFIX_UNET + '_' + parts[i])
        elif i == len(parts) - 2:
            new_parts.append(parts[i] + '_to_' + parts[i+1])
            new_parts[-1] = new_parts[-1].replace('_to_weight', '')
        elif i == len(parts) - 1:
            new_parts[-1] += '.' + parts[i]
        elif parts[i] != 'processor':
            new_parts.append(parts[i])
    new_name = '_'.join(new_parts)
    new_name = new_name.replace('__', '_')
    new_name = new_name.replace('_to_out.', '_to_out_0.')
    return new_name


def load_lora_weights(pipeline, checkpoint_path, multiplier = 1.0, device = 'cuda', dtype = torch.float16):

    # load LoRA weight from .safetensors
    if(checkpoint_path.endswith(".safetensors")):
        state_dict = load_file(checkpoint_path, device=device)
    else:
        state_dict = {}
        bin_state_dict = torch.load(checkpoint_path, map_location=device)
        for key_bin in bin_state_dict:
            key_safetensors = convert_name_to_safetensors(key_bin)
            state_dict[key_safetensors] = bin_state_dict[key_bin]
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha'] if 'alpha' in elems else None
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0
        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
    return pipeline