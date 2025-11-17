from types import ModuleType
# from src import main_dataset, main_latent_search, main_neural_latent_search, main_trainer



registry = {
    "latent_search": "main_latent_search",
    "dataset": "main_dataset",
     "trainer": "main_trainer"
}

def get_script(script_name: str) -> ModuleType:
    
    if script_name in registry:
        return registry[script_name]
     
    raise ValueError("Unknown name for main script.")
