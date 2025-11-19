from typing import Tuple, Callable, NamedTuple
from config import TrainConfig

import jax
import jax.numpy as jnp
import equinox as eqx


from vae.utils import cross_entropy_loss




class Aux(NamedTuple):
    output: jax.Array
    latent_loss: jax.Array
    progs_loss: jax.Array
    a_h_loss: jax.Array




def get_loss_fn(model_name: str) -> Callable:

    if model_name == "LeapsVAE":
        return leaps_loss_fn
    elif model_name == "StrimmedVAE":
        return strimmed_loss_fn


### NOTE: By factoring out the loss function and making it a function of 
### of the model, we can remove if ___ is not None checks!, making the code cleaner.
### for better readability, it could be that we could then reuse core logic in separate funcitons 
### to avoid code duplication.


def leaps_loss_fn(params, batch, static) -> Tuple[jax.Array, NamedTuple]:
    s_h, a_h, a_h_masks, progs, progs_masks = batch
    # The model takes in these parameters, and outputs: Teacher enforcing is enabled by default, and so was also enabled for us.
    model = eqx.combine(params, static)
    output = model(
        s_h,
        a_h,
        a_h_masks,
        progs,
        progs_masks,
    )
    # These.

    (   mu,
        sigma,
        pred_progs,
        pred_progs_logits,
        pred_progs_masks,
        pred_a_h,
        pred_a_h_logits,
        pred_a_h_masks,
    ) = output

    
    # Combine first 2 dimensions of a_h (batch_size and demos_per_program)
    a_h = jnp.reshape(a_h, (-1, a_h.shape[-1]))#.view(-1, a_h.shape[-1])
    a_h_masks = jnp.reshape(a_h_masks, (-1, a_h_masks.shape[-1]))#.view(-1, a_h_masks.shape[-1])

    # Skip first token in ground truth sequences
    a_h = a_h[:, 1:]
    a_h_masks = a_h_masks[:, 1:]

    # Flatten everything for loss calculation
    a_h_flat = jnp.reshape(a_h, (-1, 1))
    a_h_masks_flat = jnp.reshape(a_h_masks, (-1, 1))

    # Skip first token in ground truth sequences
    progs = progs[:, 1:]
    progs_masks = progs_masks[:, 1:]

    # Flatten everything for loss calculation
    progs_flat = jnp.reshape(progs, (-1, 1))
    progs_masks_flat = jnp.reshape(progs_masks, (-1, 1))

    
    pred_progs_logits = jnp.reshape(pred_progs_logits, (-1, pred_progs_logits.shape[-1])) #.view(-1, pred_progs_logits.shape[-1])
    pred_progs_masks_flat = jnp.reshape(pred_progs_masks, (-1, 1))#.view(-1, 1)
    # We combine masks here to penalize predictions that are larger than ground truth
    progs_masks_flat_combined = jnp.squeeze(jnp.maximum(
        progs_masks_flat, pred_progs_masks_flat
    ))

    
    pred_a_h_logits = jnp.reshape(pred_a_h_logits, (-1, pred_a_h_logits.shape[-1]))#.view(-1, pred_a_h_logits.shape[-1])
    pred_a_h_masks_flat = jnp.reshape(pred_a_h_masks, (-1 , 1))#.view(-1, 1)
    # We combine masks here to penalize predictions that are larger than ground truth
    a_h_masks_flat_combined = jnp.squeeze(jnp.maximum(
        a_h_masks_flat, pred_a_h_masks_flat
    ))

    # Calculate classification loss only on tokens in mask
    zero_array = jnp.array([0.0])
    progs_loss, a_h_loss = zero_array, zero_array


    ### In order for jit to work, I need to change these masks in a way that does not use advances indexing, but instead uses the masking as a computation that does not change the shape of the 
    ### input dynamically.
    mask = progs_masks_flat_combined.astype(jnp.float32)

    ### So we want the mask to make the logits of the things that are masked ineffective. 
    ### Where does the mask come from?
    # jax.debug.breakpoint()
    progs_loss = cross_entropy_loss(
        pred_progs_logits,
        jnp.reshape((progs_flat), -1),
        mask
    )

    a_h_masks_flat_combined = a_h_masks_flat_combined.astype(jnp.float32)
    a_h_loss = cross_entropy_loss(
        pred_a_h_logits,
        jnp.reshape(a_h_flat, -1),
        a_h_masks_flat_combined
    )

    latent_loss = model.get_latent_loss(mu, sigma)

    ###TODO: we just remove the programmatic loss
    total_loss = (
        TrainConfig.prog_loss_coeff * progs_loss
        + TrainConfig.a_h_loss_coeff * a_h_loss
        + TrainConfig.latent_loss_coeff * latent_loss
    )

    return total_loss, Aux(output=output, latent_loss=latent_loss, progs_loss=progs_loss, a_h_loss=a_h_loss)



def strimmed_loss_fn(params, batch, static) -> Tuple[jax.Array, NamedTuple]:
    s_h, a_h, a_h_masks, progs, progs_masks = batch
    # The model takes in these parameters, and outputs: Teacher enforcing is enabled by default, and so was also enabled for us.
    model = eqx.combine(params, static)
    output = model(
        s_h,
        a_h,
        a_h_masks,
        progs,
        progs_masks,
    )
    # These.

    (   mu,
        sigma,
        _,
        _,
        _,
        pred_a_h,
        pred_a_h_logits,
        pred_a_h_masks,
    ) = output

    
    # Combine first 2 dimensions of a_h (batch_size and demos_per_program)
    a_h = jnp.reshape(a_h, (-1, a_h.shape[-1]))#.view(-1, a_h.shape[-1])
    a_h_masks = jnp.reshape(a_h_masks, (-1, a_h_masks.shape[-1]))#.view(-1, a_h_masks.shape[-1])

    # Skip first token in ground truth sequences
    a_h = a_h[:, 1:]
    a_h_masks = a_h_masks[:, 1:]

    # Flatten everything for loss calculation
    a_h_flat = jnp.reshape(a_h, (-1, 1))
    a_h_masks_flat = jnp.reshape(a_h_masks, (-1, 1))

    # Skip first token in ground truth sequences
    progs = progs[:, 1:]
    progs_masks = progs_masks[:, 1:]

    # Flatten everything for loss calculation

    pred_a_h_logits = jnp.reshape(pred_a_h_logits, (-1, pred_a_h_logits.shape[-1]))#.view(-1, pred_a_h_logits.shape[-1])
    pred_a_h_masks_flat = jnp.reshape(pred_a_h_masks, (-1 , 1))#.view(-1, 1)
    # We combine masks here to penalize predictions that are larger than ground truth
    a_h_masks_flat_combined = jnp.squeeze(jnp.maximum(
        a_h_masks_flat, pred_a_h_masks_flat
    ))

    # Calculate classification loss only on tokens in mask
    zero_array = jnp.array([0.0])
    progs_loss, a_h_loss = zero_array, zero_array


    ### In order for jit to work, I need to change these masks in a way that does not use advances indexing, but instead uses the masking as a computation that does not change the shape of the 
    ### input dynamically.

    ### So we want the mask to make the logits of the things that are masked ineffective. 
    ### Where does the mask come from?
    # jax.debug.breakpoint()

    a_h_masks_flat_combined = a_h_masks_flat_combined.astype(jnp.float32)
    a_h_loss = cross_entropy_loss(
        pred_a_h_logits,
        jnp.reshape(a_h_flat, -1),
        a_h_masks_flat_combined
    )

    latent_loss = model.get_latent_loss(mu, sigma)

    ###TODO: we just remove the programmatic loss
    total_loss = (
        TrainConfig.prog_loss_coeff * progs_loss
        + TrainConfig.a_h_loss_coeff * a_h_loss
        + TrainConfig.latent_loss_coeff * latent_loss
    )

    # jax.debug.breakpoint() 

    return total_loss, Aux(output=output, latent_loss=latent_loss, progs_loss=progs_loss, a_h_loss=a_h_loss)
