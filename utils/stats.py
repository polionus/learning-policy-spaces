from typing import Callable


def get_loss_fn(model_name: str) -> Callable:

    if model_name == "LeapsVAE":
        return  None
    elif model_name == "StrimmedVAE":
        return None