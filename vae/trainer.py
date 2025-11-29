import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from typing import Tuple
from utils.losses import get_loss_fn, Batch
from functools import partial
from aim import Run
from logger.logger import logger
from utils.time import get_time_stamp

from torch.utils.data import DataLoader
from config import TrainConfig
from typing import NamedTuple

from .models.base_vae import BaseVAE


inner_axes = Batch(
    s_h=0, 
    a_h=0, 
    a_h_masks=0, 
    progs=None,      # <--- Broadcast: Use the same prog for all 10 steps
    progs_masks=None # <--- Broadcast
)

# OUTER AXES: Handling the '256' dimension
# We map axis 0 of everything. This peels off the '256' layer, 
# passing a [10, ...] chunk to the inner function.
outer_axes = Batch(
    s_h=0, 
    a_h=0, 
    a_h_masks=0, 
    progs=0,         # <--- Vectorize: Pick the specific prog for this batch item
    progs_masks=0    # <--- Vectorize
)

@eqx.filter_jit
@eqx.filter_vmap(in_axes=((None, None, None, None, outer_axes)))
@eqx.filter_vmap(in_axes=(None, None, None, None, inner_axes))
def train_step(params, 
               opt_state, 
               optimizer: optax.GradientTransformation, 
               loss_fn:callable, 
               batch: jax.Array) -> Tuple[eqx.Module, Tuple, jax.Array, NamedTuple]:
        

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params = params)
        params = eqx.apply_updates(params, updates)

        return params, opt_state, loss, aux

def init_training(init_params, lr: float):

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(init_params)
        return optimizer, opt_state

###TODO: make note of all things that can be passed to the trainer as a experiment description
class EpochReturn(NamedTuple):
    mean_total_loss: float
    mean_progs_loss: float
    mean_a_h_loss: float
    mean_latent_loss: float
    mean_progs_t_accuracy: float
    mean_progs_s_accuracy: float
    mean_a_h_t_accuracy: float
    mean_a_h_s_accuracy: float

class Trainer:

    def __init__(self, 
                 model: BaseVAE, 
                 run: Run,
                 save: bool
                 ):

        ###TODO: Find a cleaner way of doing this: this seems to be biting us in the ass.
        self.init_params, self.static = eqx.partition(model, eqx.is_inexact_array)
        self.run = run
        self.save = save
        ##Check if necessary?
        # self.output_dir = os.path.join("index", Config.experiment_name)
 
        
        self.loss_fn = partial(get_loss_fn(TrainConfig.model_name), static = self.static)
        # self.save_each_epoch = Config.trainer_save_params_each_epoch
        self.num_epochs = TrainConfig.num_epochs
        
        # os.makedirs(os.path.join(self.output_dir, "model"), exist_ok=True)

    ### TODO: MAke sure to also get this based on model name.
    def caluculate_run_statistics(self, 
                                  progs: jax.Array, 
                                  pred_progs: jax.Array, 
                                  progs_masks: jax.Array, 
                                  pred_progs_masks: jax.Array, 
                                  a_h: jax.Array,
                                  pred_a_h: jax.Array,
                                  a_h_masks: jax.Array,
                                  pred_a_h_masks: jax.Array,
                                  ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        
        zero_array = jnp.array([0.0])
        progs_s_accuracy, progs_t_accuracy = zero_array, zero_array


        ### These decisions are made at the choose loss function level.
        if pred_progs is not None:
            progs_masks_combined = jnp.max(progs_masks, pred_progs_masks)
            progs_t_accuracy = jnp.mean((pred_progs[progs_masks_combined] == progs[progs_masks_combined]).astype(jnp.float32))
            progs_s_accuracy = (
                jnp.mean(jnp.all((progs == pred_progs), axis = 1).astype(jnp.float32))#
            )

        a_h_s_accuracy, a_h_t_accuracy = zero_array, zero_array
        if pred_a_h is not None:
            a_h_masks_combined = jnp.max(a_h_masks, pred_a_h_masks)
            a_h_t_accuracy = jnp.mean((pred_a_h[a_h_masks_combined] == a_h[a_h_masks_combined]).astype(jnp.float32))
            a_h_s_accuracy = jnp.mean(jnp.all((a_h == pred_a_h), axis = 1))

        return progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy


    def _run_batch(self, params, batch: list, opt_state, optimizer, training=True) -> Tuple:
        
        # The model takes in these parameters, and outputs: Teacher enforcing is enabled by default, and so was also enabled for us.
        params, opt_state, total_loss, aux =  train_step(params, opt_state, optimizer, self.loss_fn, batch)
        logger.info(f'Batch done, loss: {total_loss}')
        
        self.run.track(float(total_loss), name = "Loss")
        self.run.track(float(aux.latent_loss), name = "Latent loss")
        self.run.track(float(aux.progs_loss), name = "Progs Loss")
        self.run.track(float(aux.a_h_loss), name = "A_H Loss")
        
        metrics = 0
        return params, metrics

    def _run_epoch(
        self, params, dataloader: DataLoader, epoch: int, opt_state, optimizer: optax.GradientTransformation, training=True
    ) -> EpochReturn:
        batch_info_list = jnp.zeros((len(dataloader), 8))

        for batch_idx, batch in enumerate(dataloader):
            params, batch_info = self._run_batch(params, batch, opt_state, optimizer, training)

        epoch_info_list = jnp.mean(batch_info_list, axis=0)

        return params, EpochReturn(*epoch_info_list.tolist())

    def save_run(self, params: eqx.Module):

        path = f"artifacts/params/{get_time_stamp()}.pkl"
        eqx.tree_serialise_leaves(path, params)

        logger.info(f"Model saved to {path}")


    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):

        ### NOTE: you need to make sure that your optimizer is setup:
        
        optimizer, opt_state = init_training(self.init_params, lr = TrainConfig.learning_rate)
        params = self.init_params
        
        ###NOTE:  Training loop
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"Epoch:{epoch}")
            params, train_info = self._run_epoch(params, train_dataloader, epoch, opt_state, optimizer, True)
        
        if self.save:
            self.save_run(params)