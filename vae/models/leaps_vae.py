import jax
import jax.numpy as jnp

import equinox as eqx
from utils.rng import make_key_gen
from utils.losses import Batch

from dsl import DSL
from typing import Tuple

from ..utils import init_gru, init_gru_cell, init_linear, GRU
from .base_vae import BaseVAE, ModelOutput
from karel.world import WorldState
from config import TrainConfig



###NOTE: Something confusing is that the logic that executes the policy is different from the logic that is used in training?
### So there is the policy executor method in leaps vae, there is also the execute policy method used during training

### NOTE: Using the for loops is complicating the compilation, should replace with jax.lax.scan
### TODO: Use orbax for checkpointing.


### NOTE: Where am I inputting the task?


def make_leaps_vae(progs_teacher_enforcing: bool, a_h_teacher_enforcing: bool):
    class LeapsVAE(BaseVAE):

        encoder_gru: GRU
        decoder_gru: eqx.nn.GRUCell
        decoder_mlp: eqx.nn.Sequential
        policy_gru: eqx.nn.GRUCell

        policy_mlp: eqx.nn.Sequential

        def __init__(self, dsl: DSL, hidden_size: int, key):
            assert key is not None
            super().__init__(dsl, hidden_size, key)

            self.keygen = make_key_gen(key)
            
            self.encoder_gru = init_gru(GRU(self.num_program_tokens, self.hidden_size, key = next(self.keygen)), key = next(self.keygen))

            ### The decoder only does a single pass through a single token.
            self.decoder_gru = init_gru_cell(eqx.nn.GRUCell(self.num_program_tokens + self.hidden_size, self.hidden_size, key = next(self.keygen)), key = next(self.keygen))
            

            self.decoder_mlp = eqx.nn.Sequential(
            [  
                init_linear(eqx.nn.Linear(
                        2 * self.hidden_size + self.num_program_tokens, self.hidden_size, key = next(self.keygen)), key = next(self.keygen)),
                eqx.nn.Lambda(jax.nn.tanh),
                init_linear(eqx.nn.Linear(self.hidden_size, self.num_program_tokens, key = next(self.keygen)), key = next(self.keygen)),
            ]
            )

            self.policy_gru = init_gru_cell(eqx.nn.GRUCell(
                2 * self.hidden_size + self.num_agent_actions, self.hidden_size, key = next(self.keygen)), key = next(self.keygen))
       
            self.policy_mlp = eqx.nn.Sequential(

                [
                    init_linear(eqx.nn.Linear(self.hidden_size, self.hidden_size, key = next(self.keygen)), key = next(self.keygen)),
                    eqx.nn.Lambda(jax.nn.tanh),
                    init_linear(eqx.nn.Linear(self.hidden_size, self.hidden_size, key = next(self.keygen)), key = next(self.keygen)),
                    eqx.nn.Lambda(jax.nn.tanh),
                    init_linear(eqx.nn.Linear(self.hidden_size, self.num_agent_actions, key = next(self.keygen)), key = next(self.keygen)),
                ]
            )
        
        @eqx.filter_vmap(in_axes=(None, 0, 0))
        def encode(self, progs: jax.Array, progs_mask: jax.Array):

            """Encode a batch of programs and program masks. """
        
            enc_prog = self.vectorized_token_encoder(progs)
            enc_hidden_state = self.encoder_gru((enc_prog, progs_mask), jnp.zeros(self.hidden_size))
            z, mu, sigma = self.sample_latent_vector(enc_hidden_state)
            
            return z, mu, sigma            

        @eqx.filter_vmap(in_axes=(None, 0, 0))
        def decode(self, z: jax.Array, progs: jax.Array):

            # batch_size, _ = z.shape
            gru_hidden_state = z
            current_token = jnp.array(0, dtype = jnp.int32)

            grammar_state = self.syntax_checker.get_initial_checker_state()
            init_state = current_token, gru_hidden_state, grammar_state

            def decode_step(state, iter):
            

                current_tokens, gru_hidden_state, grammar_state = state
                token_embedding = self.token_encoder(current_tokens)
                gru_inputs = jnp.concatenate([token_embedding, z], axis= -1)
                gru_hidden_state = self.decoder_gru(gru_inputs, gru_hidden_state)
                mlp_input = jnp.concatenate([gru_hidden_state, token_embedding, z])
                
                pred_token_logits = self.decoder_mlp(mlp_input)
                # syntax_mask, grammar_state = self.get_syntax_mask(current_tokens, grammar_state)
                # pred_token_logits += syntax_mask
            
                ### Why take soft max and then immediately argmax?
                pred_tokens = jnp.argmax(self.softmax(pred_token_logits), axis = -1)
            
                if progs_teacher_enforcing:
                    current_token = progs[iter].astype(jnp.int32)
                else:
                    current_token = pred_tokens

                new_state = current_token, gru_hidden_state, grammar_state
                y = (pred_tokens, pred_token_logits)

                return new_state, y
            
            final_state, ys = jax.lax.scan(decode_step, init_state, jnp.arange(1, self.max_program_length))
            pred_progs, pred_progs_logits = ys    

            
            # pred_progs = jnp.swapaxes(pred_progs, 0, 1)
            # pred_progs_logits = jnp.swapaxes(pred_progs_logits, 0, 1)
            pred_progs_masks = pred_progs != self.num_program_tokens -1 ## Mask Over the Pad Tokens. 


            return pred_progs, pred_progs_logits, pred_progs_masks


        # There are two compositions of the vmap since the a_h and s_h inputs have 2 batch dimensions (batch_dim, num_demos_pre_prog)
        @eqx.filter_vmap(in_axes=(None, None, 1, 1))
        @eqx.filter_vmap(in_axes=(None, 0, 0, 0))
        def policy_executor(self, 
                            z: jax.Array, 
                            s_h: jax.Array, 
                            a_h: jax.Array):          
            _, c, h, w = s_h.shape
            current_state = s_h[0, :, :, :]
            current_action = (self.num_agent_actions - 1) 
        
            gru_hidden = z
            world = -1

            ### This should work with vmap
            if not a_h_teacher_enforcing:
                world = self.env_init(current_state)

            terminated_policy = jnp.zeros_like(current_action, dtype = jnp.bool)
            mask_valid_actions = jnp.array((self.num_agent_actions - 1) * [-jnp.finfo(dtype = jnp.float32).max] + [0.0])

            init_state = current_state, current_action, gru_hidden, terminated_policy, world

            def policy_executor_step(state, iter: int):
                
                current_state, current_action, gru_hidden, terminated_policy, world = state
                enc_state = self.state_encoder(current_state)
                enc_action = self.action_encoder(current_action)
                gru_inputs = jnp.concatenate([z, enc_state, enc_action])
                gru_hidden = self.policy_gru(gru_inputs, gru_hidden) ### This is so slow!!!!!!
                pred_action_logits = self.policy_mlp(gru_hidden) ### this is also slow, but not as slow.
        
                masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
                
                
                if a_h_teacher_enforcing:
                    ##This does not chagne the type of the input
                    current_state = s_h[iter, :, :, :]
                    current_action = a_h[iter].astype(jnp.int32)
                
                else:
                   
                    current_action = jnp.argmax(masked_action_logits)
                    current_state, world = self.env_step(world, current_action) ### The difference this makes is minimal.
                    
                terminated_policy = jnp.logical_or(current_action == self.num_agent_actions - 1, terminated_policy)

                new_state = (current_state, current_action, gru_hidden, terminated_policy, world)
                y = (current_action,pred_action_logits)
        
                return new_state, y
            
            final_state, ys = jax.lax.scan(policy_executor_step, init_state, jnp.arange(1, self.max_demo_length))
            pred_a_h, pred_a_h_logits = ys  
            pred_a_h_masks = pred_a_h != self.num_agent_actions - 1

            return pred_a_h, pred_a_h_logits, pred_a_h_masks, world
        

        def __call__(self, batch:Batch):
            
            
            z, mu, sigma = self.encode(batch.progs, batch.progs_masks) ## This part is fine.

            ### The methods have been constructed at initialization time, and hence do not need the flags passed in 
            # ### for each call.

            
            ### I might have to change this to a pytree with eqx.Module
            decoder_result = self.decode(z, batch.progs) ### This makes it slower, but not as slow. 
            pred_progs, pred_progs_logits, pred_progs_masks = decoder_result

            
            # ### The methods have been constructed at initialization time, and hence do not need the flags passed in 
            # ### for each call. These functions need their signatures fixed so that they work properly (I have removed the teacher_enforcing flags.)
            policy_result = self.policy_executor(z, batch.s_h, batch.a_h) ### BUG: This is the slowest part.
            pred_a_h, pred_a_h_logits, pred_a_h_masks, world = policy_result ## Is the world state going to be necessary (I think not?)
         
        
            return ModelOutput(mu,
                            sigma,
                            pred_progs,
                            pred_progs_logits,
                            pred_progs_masks,
                            pred_a_h,
                            pred_a_h_logits,
                            pred_a_h_masks)
        
        def encode_program_to_latent(self, prog: jax.Array):
            if prog.ndim == 1:
                prog = jnp.expand_dims(prog, axis = 0)

            ##Why is this mask used?
            prog_mask = prog != self.num_program_tokens - 1

            z = self.encode(prog, prog_mask)

            return z
        

        def decode_vector(self, z:jax.Array) -> jax.Array:
            '''It receives a population of latent programs and then outputs the decoded programs'''

            # I separated this block to able to jit it.
            # The reason we cannot return the padded sequences is that the interpreter doesn't accept the pad token?
            pred_progs, _, pred_progs_masks = eqx.filter_jit(self.decode)(z, None)
            pred_progs_masks = pred_progs_masks.astype(jnp.bool)

            pred_progs_tokens = []
            for prog, prog_mask in zip(pred_progs, pred_progs_masks):

                # Add the DEF (0) token to all programs.
                pred_progs_tokens.append([0] + (prog[prog_mask]).tolist())

            return pred_progs_tokens
    return LeapsVAE

LeapsVAE = make_leaps_vae(TrainConfig.prog_teacher_enforcing, TrainConfig.a_h_teacher_enforcing)
LeapsVAESearch = make_leaps_vae(False, False)