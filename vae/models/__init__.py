from typing import Union
from .base_vae import BaseVAE
from .leaps_vae import LeapsVAE
from .strimmed_vae import StrimmedVAE


from dsl import DSL
import jax
from config import Config

key = jax.random.key(Config.model_seed)


def load_model(
    model_cls_name: str,
    dsl: DSL,
    hidden_size: Union[None, int] = None,
) -> BaseVAE:
    # return
    model_cls = globals()[model_cls_name]
    # assert issubclass(model_cls, BaseVAE)
    return model_cls(dsl, hidden_size, key)
