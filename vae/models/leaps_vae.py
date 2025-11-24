import jax
import jax.numpy as jnp

import equinox as eqx
from utils.rng import make_key_gen

from dsl import DSL
from typing import Tuple

from ..utils import init_gru, init_gru_cell, init_linear, GRU
from .base_vae import BaseVAE, ModelOutput
from karel.world import WorldState
from config import TrainConfig

WORLD_IN_AXES = WorldState(
    numAPICalls=0,      # scalar, shared across the batch
    crashed=0,          # scalar, shared
    s=0,                   # array, batched along axis 0
    hero_pos=0,            # array, batched along axis 0
    markers_grid=0,        # array, batched along axis 0
    terminated=0        # scalar, shared
)

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

        batched_enc_state: callable
        batched_enc_action: callable
        batched_policy_gru: callable
        batched_policy_mlp: callable
        batched_env_step: callable      


        batch_token_encoder: callable 
        batch_decoder_gru: callable 
        batch_decoder_mlp: callable 
        batch_get_syntax_mask: callable 


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


            self.batch_token_encoder = eqx.filter_vmap(self.token_encoder)
            self.batch_decoder_gru = eqx.filter_vmap(self.decoder_gru, in_axes = (0, 0))
            self.batch_decoder_mlp = eqx.filter_vmap(self.decoder_mlp)
            self.batch_get_syntax_mask = eqx.filter_vmap(self.get_syntax_mask, in_axes=(0, 0))


            ### batched stuff for the policy executor
            self.batched_enc_state = eqx.filter_vmap(self.state_encoder)
            self.batched_enc_action = eqx.filter_vmap(self.action_encoder)
            self.batched_policy_gru = eqx.filter_vmap(self.policy_gru, in_axes = (0, 0))
            self.batched_policy_mlp = eqx.filter_vmap(self.policy_mlp)
            self.batched_env_step = eqx.filter_vmap(self.env_step, in_axes=(WORLD_IN_AXES, 0, 0))
        ### Strategy: Create static pure function generators, that take in the static argnums and generate the 
        ### desired function.

        def encode(self, progs: jax.Array, progs_mask: jax.Array):

            """Encode a batch of programs and program masks. """

            ## Perhaps I would have to deal with shapes later.

            # batch_size, demos_per_program = progs.shape
            # progs = progs.reshape((batch_size * demos_per_program, -1))
            # progs_mask = progs_mask.reshape((batch_size * demos_per_program, -1))
            ### TODO: extract the vmap function and simply call it here instead of redfining.
            enc_progs = eqx.filter_vmap(lambda row: eqx.filter_vmap(self.token_encoder)(row.astype(jnp.int32)))(progs)
            enc_hidden_state = eqx.filter_vmap(self.encoder_gru, in_axes = (0, None))((enc_progs, progs_mask), jnp.zeros(self.hidden_size))
        
            z, mu, sigma = self.sample_latent_vector(enc_hidden_state)

            return z, mu, sigma            

        def decode(self, z: jax.Array, progs: jax.Array):
            
            batch_size, _ = z.shape
            gru_hidden_state = z
            current_tokens = jnp.zeros((batch_size), dtype = jnp.int32)  
            grammar_state = [
                self.syntax_checker.get_initial_checker_state() for _ in range(batch_size)
            ]

            ### Stack bathc dimension in PyTree Grammar states
            grammar_state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *grammar_state)
        
            init_state = current_tokens, gru_hidden_state, grammar_state

            def decode_step(state, iter):
            
            ### Our first attempt at jitting this succeeds, even though this has an if statement inside it!
            ### This is because a class factory is used.

                current_tokens, gru_hidden_state, grammar_state = state

                token_embedding = self.batch_token_encoder(current_tokens)
                gru_inputs = jnp.concatenate([token_embedding, z], axis= -1)
                gru_hidden_state = self.batch_decoder_gru(gru_inputs, gru_hidden_state)

                
                mlp_input = jnp.concatenate([gru_hidden_state, token_embedding, z], axis= 1)
                pred_token_logits = self.batch_decoder_mlp(mlp_input)
            
                syntax_mask, grammar_state = self.batch_get_syntax_mask(current_tokens, grammar_state)
                pred_token_logits += syntax_mask
            
                ### Why take soft max and then immediately argmax?
                pred_tokens = jnp.argmax(self.softmax(pred_token_logits), axis = -1)
            
                if progs_teacher_enforcing:
                    current_tokens = jnp.reshape(progs[:, iter], batch_size).astype(jnp.int32)
                else:
                    current_tokens = jnp.reshape(pred_tokens, batch_size)

                new_state = current_tokens, gru_hidden_state, grammar_state
                y = (pred_tokens, pred_token_logits)

                return new_state, y
            
            final_state, ys = jax.lax.scan(decode_step, init_state, jnp.arange(1, self.max_program_length))
            pred_progs, pred_progs_logits = ys    

            pred_progs = jnp.swapaxes(pred_progs, 0, 1)
            pred_progs_logits = jnp.swapaxes(pred_progs_logits, 0, 1)
            pred_progs_masks = pred_progs != self.num_program_tokens -1 ## Mask Over the Pad Tokens. 


            return pred_progs, pred_progs_logits, pred_progs_masks

        def policy_executor(self, 
                            z: jax.Array, 
                            s_h: jax.Array, 
                            a_h: jax.Array,
                            a_h_mask: jax.Array,
                            ):
            
            
            batch_size, demos_per_program, _, c, h, w = s_h.shape
            current_state = jnp.reshape(s_h[:, :, 0, :, :, :], (batch_size * demos_per_program, c, h, w))
            ones = jnp.ones((batch_size * demos_per_program, 1), dtype=jnp.int32)
            ## Why is this step necessary? ## Why do we use an action encoder, that is an embedding?
            current_action = (self.num_agent_actions - 1) * ones
            
            ##This is not changed by the policy executor step.
            
            z_repeated = jnp.tile(jnp.expand_dims(z, axis = 1), (1, demos_per_program, 1))
            z_repeated = jnp.reshape(z_repeated, (batch_size * demos_per_program, self.hidden_size))

            # gru_hidden = jnp.expand_dims(z_repeated, axis = 0)
            gru_hidden = z_repeated
            world = -1
            ### Is it possible to do the env_init without triggering compilation?
            
            ### there is a problem: if teacher enforcing is off, then world = -1, 

            
            if not a_h_teacher_enforcing:
                ### this block needs to be jitted 
                world = self.env_init(current_state)

            terminated_policy = jnp.zeros_like(current_action, dtype = jnp.bool)

            ## What is this magic?
            mask_valid_actions = jnp.array((self.num_agent_actions - 1) * [-jnp.finfo(dtype = jnp.float32).max] + [0.0])

            init_state = current_state, current_action, gru_hidden, terminated_policy, world
            # jax.debug.breakpoint()
            
            ### batch the world: world


            ##### HERE #####
            ### Create a step function and jit the body.
            ### what are the states, and the iterables?
            ### We are going to iterate over jnp.arange(1, self.max_demo_length)
            ### Everything else is part of the state

            ### By adding this, we can simplify the inputs to the problem: 
            ### Do I need to input s_h?

            def policy_executor_step(state, iter: int):
                
                current_state, current_action, gru_hidden, terminated_policy, world = state
                enc_state = self.batched_enc_state(current_state)
                enc_action = self.batched_enc_action(jnp.squeeze(current_action, axis = -1))

                gru_inputs = jnp.concatenate([z_repeated, enc_state, enc_action], axis = -1)

                gru_hidden = self.batched_policy_gru(gru_inputs, gru_hidden) ### This is so slow!!!!!!
                ### TODO: Compare this to the concatenation done in the decoder
                pred_action_logits = self.batched_policy_mlp(gru_hidden) ### this is also slow, but not as slow.
                # pred_action_logits = jnp.ones((128, self.num_agent_actions)) * jax.random.normal(key = jax.random.key(iter), shape = (128, self.num_agent_actions)) * gru_hidden[:, :6]
             

                masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions

                # current_action = jnp.reshape(jnp.argmax(self.softmax(masked_action_logits), axis = -1), (-1, 1)) ##Having both argmax and softmax unnecessary?
                current_action = jnp.reshape(jnp.argmax(masked_action_logits, axis = -1), (-1, 1))
                ###TODO: for readability, make the size changes more semantic, so that the logic is also separated, like the reshapes or self.num_agent_actions -1 (Final action, NULL action)
                ### Something is wrong with the shape without teacher enforcing.
                ## BUG: There is a bug when teacher enforcing is on
                if a_h_teacher_enforcing:

                    ##This does not chagne the type of the input
                    current_state = jnp.reshape(s_h[:, :, iter, :, :, :], (batch_size * demos_per_program, c, h, w))
                    current_action = jnp.reshape(a_h[:, :, iter], (batch_size * demos_per_program, 1)).astype(jnp.int32)

                else:
                    ### TODO: make sure the unbatched dimensions would give the right output?
                    ### This does.
                    current_state, world = self.batched_env_step(world, current_state, current_action) ### The difference this makes is minimal.
                    
            
                terminated_policy = jnp.logical_or(current_action == self.num_agent_actions - 1, terminated_policy)
                # jax.debug.breakpoint()

                new_state = (current_state, current_action, gru_hidden, terminated_policy, world)
                y = (current_action,pred_action_logits)

                return new_state, y
            
         
            final_state, ys = jax.lax.scan(policy_executor_step, init_state, jnp.arange(1, self.max_demo_length))
            pred_a_h, pred_a_h_logits = ys  


            pred_a_h = jnp.swapaxes(pred_a_h, 0, 1)
            pred_a_h_logits = jnp.swapaxes(pred_a_h_logits, 0, 1)
            pred_a_h_masks = pred_a_h != self.num_agent_actions - 1

            return pred_a_h, pred_a_h_logits, pred_a_h_masks, world
        

        def __call__(self, s_h, a_h, a_h_mask, prog, prog_mask):
            
            z, mu, sigma = self.encode(prog, prog_mask) ## This part is fine.

            ### The methods have been constructed at initialization time, and hence do not need the flags passed in 
            # ### for each call.
            decoder_result = self.decode(z, prog) ### This makes it slower, but not as slow. 
            pred_progs, pred_progs_logits, pred_progs_masks = decoder_result

            # ### The methods have been constructed at initialization time, and hence do not need the flags passed in 
            # ### for each call. These functions need their signatures fixed so that they work properly (I have removed the teacher_enforcing flags.)
            policy_result = self.policy_executor(z, s_h, a_h, a_h_mask) ### BUG: This is the slowest part.
            pred_a_h, pred_a_h_logits, pred_a_h_masks, world = policy_result ## Is the world state going to be necessary (I think not?)
        

            ### So when I change these, 

            
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
            pred_progs, _, pred_progs_masks = self.decode(z, None, force_disable_teacher_enforcing=True)
            pred_progs_masks = pred_progs_masks.astype(jnp.bool)
            
            pred_progs_tokens = []
            for prog, prog_mask in zip(pred_progs, pred_progs_masks):

                # Add the DEF (0) token to all programs.
                pred_progs_tokens.append([0] + (prog[prog_mask]).tolist())

            return pred_progs_tokens
    return LeapsVAE

LeapsVAE = make_leaps_vae(TrainConfig.prog_teacher_enforcing, TrainConfig.a_h_teacher_enforcing)
# LeapsVAE = make_leaps_vae(False, False)








