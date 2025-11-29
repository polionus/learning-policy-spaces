from typing import NamedTuple
import jax
import jax.numpy as jnp
from functools import partial
from utils.losses import Batch

import equinox as eqx


from dsl import DSL
from dsl.syntax_checker import PySyntaxChecker, CheckerState
from karel.world import STATE_TABLE
from karel.world import make_world_state, WorldState
from karel.world_batch import step
from config import Config
from utils.rng import make_key_gen
from typing import Callable, Generator
from ..utils import init_conv2d, init_linear

### TODO: can you make sure that the vae doesn't have to have the program length?
STATE_SHAPE = (len(STATE_TABLE), Config.env_height(), Config.env_width())


      # scalar, shared


#### TODO: Maybe these named tuples need to be jax arrays.
class ModelOutput(NamedTuple):

    mu: jax.Array
    sigma: jax.Array
    pred_progs: jax.Array
    pred_progs_logits: jax.Array
    pred_progs_masks: jax.Array
    pred_a_h: jax.Array
    pred_a_h_logits: jax.Array
    pred_a_h_masks: jax.Array

### Yay, we have converted all important for loops and if statements into jax equivalents!
class BaseVAE(eqx.Module):

    max_demo_length: int
    max_program_length: int
    hidden_size: int | None
    num_agent_actions: int
    num_program_tokens: int
    pad_token: int
    
    state_encoder: eqx.Module
    action_encoder: eqx.Module
    token_encoder: eqx.Module
    vectorized_token_encoder: eqx.Module

    encoder_mu: eqx.Module
    encoder_log_sigma: eqx.Module
    softmax: Callable
    keygen: Generator


    syntax_checker: PySyntaxChecker
    hidden_size: int
    dsl: DSL
    world_states: WorldState

    def __init__(self, dsl: DSL, hidden_size: int, key):
        super().__init__()
        
        self.world_states = -1
        

        self.dsl = dsl
        
        self.keygen = make_key_gen(key)
        
        self.max_demo_length = Config.data_max_demo_length
        self.max_program_length = Config.data_max_program_length

        if hidden_size is None:
            self.hidden_size = Config.model_hidden_size
        else:
            self.hidden_size = hidden_size

        self.num_agent_actions = (
            len(dsl.get_actions()) + 1
        )  # +1 because we have a NOP action

        # T
        self.num_program_tokens = len(
            dsl.get_tokens()
        )  # dsl includes <pad> and <HOLE> tokens

        self.pad_token = dsl.t2i["<pad>"]

        # ## fix initiatlization
        # key, key1, key2, key3 = jax.random.split(key, 4)

        self.state_encoder = eqx.nn.Sequential(
            [
                init_conv2d(eqx.nn.Conv2d(STATE_SHAPE[0], 32, 3, stride=1, key = next(self.keygen)), key= next(self.keygen)),
                eqx.nn.Lambda(jax.nn.relu),
                init_conv2d(eqx.nn.Conv2d(32, 32, 3, stride=1, key = next(self.keygen)), key = next(self.keygen)),
                eqx.nn.Lambda(jax.nn.relu), 
                eqx.nn.Lambda(jnp.ravel), 
                init_linear(eqx.nn.Linear(32 * 16, self.hidden_size, key = next(self.keygen)), key = next(self.keygen))

            ]
        )

        self.action_encoder = eqx.nn.Embedding(self.num_agent_actions, self.num_agent_actions, key=next(self.keygen)) 
        #Extra vmap over tokens (which are normally not vmapped.)
        self.token_encoder = eqx.nn.Embedding(num_embeddings=self.num_program_tokens, embedding_size=self.num_program_tokens, key = next(self.keygen))
        self.vectorized_token_encoder = eqx.filter_vmap(self.token_encoder)

        self.encoder_mu = eqx.nn.Linear(self.hidden_size, self.hidden_size, key = next(self.keygen))
        self.encoder_log_sigma = eqx.nn.Linear(self.hidden_size, self.hidden_size, key = next(self.keygen))

        self.softmax = partial(jax.nn.log_softmax, axis = -1)
        self.syntax_checker = PySyntaxChecker(dsl.t2i) ### The syntax checker is only instantiated once at the start, and is not rerun in the training loop.


    def env_init(self, states: jax.Array) -> WorldState:
        states = jnp.moveaxis(states, [-1, -2, -3], [-2, -3, -1])
        #self._world = WorldBatch(states)
        ## I have successfully created the vmapped version of the worlds.
        ## I need to see how to use them.    
        return make_world_state(states)
        
    def env_step(self, world: WorldState, actions: jax.Array):

        new_states, world = step(world, actions)
        new_states = jnp.transpose(new_states, (2, 0, 1)).astype(jnp.float32)
        
        return new_states, world
    

    def sample_latent_vector(self, enc_hidden_state: jax.Array) -> jax.Array:
        
        mu = self.encoder_mu(enc_hidden_state)
        log_sigma = self.encoder_log_sigma(enc_hidden_state)
        sigma = jnp.exp(log_sigma)
        std_z = jax.random.normal(next(self.keygen), sigma.shape)

        z = mu + sigma * std_z

        return z, mu, sigma
    
    ### I can't change the self.z_mu, so I need to pass it. 
    def get_latent_loss(self, mu, sigma):
        mean_sq = mu * mu
        stddev_sq = sigma * sigma
        return 0.5 * jnp.mean(mean_sq + stddev_sq - jnp.log(stddev_sq) - 1)

    def get_syntax_mask(
        self, current_token: jax.Array, grammar_state: CheckerState
    ):
        current_token = jnp.expand_dims(current_token, axis = 0)
        grammar_state, sequence_mask = self.syntax_checker.get_sequence_mask(grammar_state, current_token)
        syntax_mask = jnp.where(
                sequence_mask,                                # boolean mask
                -jnp.finfo(jnp.float32).max * jnp.ones_like(sequence_mask, dtype=jnp.float32),
                jnp.zeros_like(sequence_mask, dtype=jnp.float32),
                ).squeeze()

        return syntax_mask, grammar_state    


    def __call__(
        self,
        batch: Batch,   
    ):
        raise NotImplementedError


    def encode_program(self, prog: jax.Array) -> jax.Array:
        raise NotImplementedError

    def decode_vector(self, z: jax.Array) -> list[list[int]]:
        raise NotImplementedError




