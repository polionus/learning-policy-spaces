import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from typing import Tuple
from utils.losses import get_loss_fn
from functools import partial

 
from aim import Run

from torch.utils.data import DataLoader
from config import Config
from typing import NamedTuple



from .models.base_vae import BaseVAE



@eqx.filter_jit
def train_step(params, opt_state, optimizer: optax.GradientTransformation, loss_fn:callable, batch: jax.Array) -> Tuple[eqx.Module, Tuple, jax.Array, NamedTuple]:

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

    def __init__(self, model: BaseVAE, run: Run):

        ###TODO: Find a cleaner way of doing this: this seems to be biting us in the ass.
        # self.initial_model = model
        self.init_params, self.static = eqx.partition(model, eqx.is_inexact_array)
        self.run = run

        ##Check if necessary?
        self.output_dir = os.path.join("index", Config.experiment_name)
 
        
        self.loss_fn = partial(get_loss_fn(Config.model_name), static = self.static)
        self.save_each_epoch = Config.trainer_save_params_each_epoch
        self.num_epochs = Config.trainer_num_epochs #exp
        
        os.makedirs(os.path.join(self.output_dir, "model"), exist_ok=True)

    

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


        ### these conditions are here, since the output of the function we have 
        ### might be leaps vae, might be something else, 
        ### and if they are something else, they will be set to None.

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
        self.run.track(float(total_loss), name = "Loss")
        s_h, a_h, a_h_masks, progs, progs_masks = batch

        (   _, 
            _,
            pred_progs,
            _,
            pred_progs_masks,
            pred_a_h,
            _,
            pred_a_h_masks,
        ) = aux.output
        # progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy = self.caluculate_run_statistics(progs, 
        #                                                                                                     pred_progs, 
        #                                                                                                     progs_masks, 
        #                                                                                                     pred_progs_masks, 
        #                                                                                                     a_h, 
        #                                                                                                     pred_a_h, 
        #                                                                                                     a_h_masks,
        #                                                                                                     pred_a_h_masks
        #                                                                                                     )  
        progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy = 0, 0, 0, 0  
        
        # metrics = (
        #         total_loss.item(),
        #         aux.progs_loss.item(),
        #         aux.a_h_loss.item(),
        #         aux.latent_loss.item(),
        #         progs_t_accuracy.item(),
        #         progs_s_accuracy.item(),
        #         a_h_t_accuracy.item(),
        #         a_h_s_accuracy.item())
        metrics = 0
        return params, metrics

    def _run_epoch(
        self, params, dataloader: DataLoader, epoch: int, opt_state, optimizer: optax.GradientTransformation, training=True
    ) -> EpochReturn:
        batch_info_list = jnp.zeros((len(dataloader), 8))

        for batch_idx, batch in enumerate(dataloader):
            print("New Batch!")
            
            batch = tuple(jax.device_put(item.astype(np.float32, copy = False)) for item in batch)
            params, batch_info = self._run_batch(params, batch, opt_state, optimizer, training)
            
            # batch_info_list[batch_idx] = batch_info

        epoch_info_list = jnp.mean(batch_info_list, axis=0)

        return params, EpochReturn(*epoch_info_list.tolist())

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):

        ### NOTE: you need to make sure that your optimizer is setup:
        
        optimizer, opt_state = init_training(self.init_params, lr = Config.trainer_optim_lr)
        params = self.init_params
        
        

        ###NOTE:  Training loop
        for epoch in range(1, self.num_epochs + 1):
            print('New epoch!')
            params, train_info = self._run_epoch(params, train_dataloader, epoch, opt_state, optimizer, True)
            