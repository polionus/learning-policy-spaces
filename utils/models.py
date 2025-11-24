import equinox as eqx
from vae.models.base_vae import BaseVAE  # Your model class

def load_model_from_template(checkpoint_path: str, model_template: BaseVAE) -> BaseVAE:
    """
    Reconstruct a model from saved parameters.
    
    Args:
        checkpoint_path: Path to the saved .pkl file
        model_template: An instance of the model with the same architecture
                       (used to get the static parts)
    
    Returns:
        Reconstructed model with loaded parameters
    """
    # model = load_model(TrainConfig.model_name, dsl) #This loads the PyTree
    trainable_template, static = eqx.partition(model_template, eqx.is_inexact_array) #Given the Pytree, this separates the trainable and static parts. 
    #given the path, and the trainable part, we 
    model_trainable_params = eqx.tree_deserialise_leaves(checkpoint_path, trainable_template)
    model = eqx.combine(model_trainable_params, static)
    
    return model
