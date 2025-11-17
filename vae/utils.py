import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import optax
import equinox as eqx



def cross_entropy_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets.astype(jnp.int32))
    loss = loss * mask
    return loss.mean()



class GRU(eqx.Module):

    gru_cell: eqx.nn.GRUCell
    
    def __init__(self, input_size: int, hidden_size: int, use_bias = True, key: jax.Array | None = None):

        self.gru_cell = eqx.nn.GRUCell(input_size, hidden_size, use_bias=use_bias, key = key)
   
    def __call__(self, xs: jax.Array, init_state: jax.Array):
        '''This function call requires you to give the init_state explicitly, and You also need to pass the mask in the input as a tuple. 

            - The masking is to handle variable length sequences that have been padded. if the sequence has no padding, the mask should be all True.


        '''
        def step(state, input):
            seq, mask = input

            possible_new_state = self.gru_cell(seq, state)
            new_state = jnp.where(mask, possible_new_state, state)
            
            return new_state, None
        
        final_state, _ = jax.lax.scan(step, init_state, xs)
        return final_state
    


# TODO: comment below is from original LEAPS: check if necessary
# replace unmask_idx with unmask_idx2 after verify, ing identity
def unmask_idx(output_mask_all, first_end_token_idx, max_program_len):
    for p_idx in range(first_end_token_idx.shape[0]):
        t_idx = int(first_end_token_idx[p_idx].detach().cpu().numpy())
        if t_idx < max_program_len:
            output_mask_all[p_idx, t_idx] = True
    return output_mask_all.to(torch.bool)


def unmask_idx2(x):
    seq, seq_len = x
    if seq_len < seq.shape[0]:
        seq[seq_len] = True
        return True
    return False

### So the init gru function makes the bias values zero, and does orthogonal initialization for the weights.
### NOTE: When writing orthogonal intializers, when to keep in mind the gain for each initialization.
def init_gru(gru_layer: GRU, key: jax.Array):
    '''Get all the weights of a GRU layer'''

    ortho_init = jax.nn.initializers.orthogonal()
    zero_init = jax.nn.initializers.zeros
    
    ### Get all the weights: 
    gru_cell = gru_layer.gru_cell
    weight_hh_shape = gru_cell.weight_hh.shape
    weight_ih_shape = gru_cell.weight_ih.shape

    bias_shape = gru_cell.bias.shape
    bias_n_shape = gru_cell.bias_n.shape

    key, subkey, subkey1, subkey2 = jax.random.split(key, 4)

    new_weight_hh = ortho_init(key, weight_hh_shape)
    new_weight_ih = ortho_init(subkey, weight_ih_shape)


    new_bias_n = zero_init(subkey1, bias_n_shape)
    new_bias = zero_init(subkey2, bias_shape)

    def get_bias(gru_layer): return gru_layer.gru_cell.bias
    def get_bias_n(gru_layer): return gru_layer.gru_cell.bias_n

    def get_weight_hh(gru_layer): return gru_layer.gru_cell.weight_hh
    def get_weight_ih(gru_layer): return gru_layer.gru_cell.weight_ih


    new_gru = eqx.tree_at(get_bias, gru_layer, new_bias)
    new_gru = eqx.tree_at(get_bias_n, new_gru, new_bias_n)
    new_gru = eqx.tree_at(get_weight_hh, new_gru, new_weight_hh)
    new_gru = eqx.tree_at(get_weight_ih, new_gru, new_weight_ih)


    return new_gru


def init_gru_cell(gru_cell: eqx.nn.GRUCell, key: jax.Array):
    '''Get all the weights of a GRU layer'''

    ortho_init = jax.nn.initializers.orthogonal()
    zero_init = jax.nn.initializers.zeros
    
    ### Get all the weights: 
    weight_hh_shape = gru_cell.weight_hh.shape
    weight_ih_shape = gru_cell.weight_ih.shape

    bias_shape = gru_cell.bias.shape
    bias_n_shape = gru_cell.bias_n.shape

    key, subkey, subkey1, subkey2 = jax.random.split(key, 4)

    new_weight_hh = ortho_init(key, weight_hh_shape)
    new_weight_ih = ortho_init(subkey, weight_ih_shape)


    new_bias_n = zero_init(subkey1, bias_n_shape)
    new_bias = zero_init(subkey2, bias_shape)

    def get_bias(gru_cell): return gru_cell.bias
    def get_bias_n(gru_cell): return gru_cell.bias_n

    def get_weight_hh(gru_cell): return gru_cell.weight_hh
    def get_weight_ih(gru_cell): return gru_cell.weight_ih


    new_gru_cell = eqx.tree_at(get_bias, gru_cell, new_bias)
    new_gru_cell = eqx.tree_at(get_bias_n, new_gru_cell, new_bias_n)
    new_gru_cell = eqx.tree_at(get_weight_hh, new_gru_cell, new_weight_hh)
    new_gru_cell = eqx.tree_at(get_weight_ih, new_gru_cell, new_weight_ih)


    return new_gru_cell


def init_linear(linear_layer: eqx.nn.Linear, key: jax.Array):

    ortho_init = jax.nn.initializers.orthogonal()
    zero_init = jax.nn.initializers.zeros
    
    ### Get all the weights: 
    
    weight_shape = linear_layer.weight.shape
    bias_shape = linear_layer.bias.shape

    key, subkey = jax.random.split(key)

    new_weight = ortho_init(key, weight_shape)
    new_bias = zero_init(subkey, bias_shape)


    def get_weight(linear_layer): return linear_layer.weight
    def get_bias(linear_layer): return linear_layer.bias

    new_layer = eqx.tree_at(get_bias, linear_layer, new_bias)
    new_layer = eqx.tree_at(get_weight, new_layer, new_weight)

    return new_layer



def init_conv2d(conv_layer: eqx.nn.Linear, key: jax.Array):

    key, subkey, subkey1 = jax.random.split(key, 3)

    ortho_init = jax.nn.initializers.orthogonal()
    zero_init = jax.nn.initializers.zeros
    
    ### Get all the weights: 
    
    weight_shape = conv_layer.weight.shape
    bias_shape = conv_layer.bias.shape

    key, subkey = jax.random.split(key)

    new_weight = ortho_init(key, weight_shape)
    new_bias = zero_init(subkey, bias_shape)


    def get_weight(conv_layer): return conv_layer.weight
    def get_bias(conv_layer): return conv_layer.bias

    new_layer = eqx.tree_at(get_bias, conv_layer, new_bias)
    new_layer = eqx.tree_at(get_weight, new_layer, new_weight)

    return new_layer


    


# def init_gru(module: torch.nn.GRU):
    
#     for name, param in module.named_parameters():
#         if "bias" in name:
#             nn.init.constant_(param, 0)
#         elif "weight" in name:
#             nn.init.orthogonal_(param)


# def init(module, weight_init, bias_init, gain=1.0):
#     weight_init(module.weight.data, gain=gain)
#     bias_init(module.bias.data)
#     return module


def masked_mean(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim) / torch.sum(
        mask, dim=dim, keepdim=keepdim
    )


def masked_sum(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim)


def add_record(key, value, global_logs):
    if "logs" not in global_logs["info"]:
        global_logs["info"]["logs"] = {}
    logs = global_logs["info"]["logs"]
    split_path = key.split(".")
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


def log_record_dict(usage, log_dict, global_logs):
    for log_key, value in log_dict.items():
        add_record("{}.{}".format(usage, log_key), value, global_logs)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
