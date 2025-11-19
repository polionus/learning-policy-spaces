import os 
import sys
from aim import Run
from logger.logger import logger


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dsl import DSL
from vae.models import load_model
from vae.program_dataset import make_dataloaders
from vae.trainer import Trainer
from config import TrainConfig
import tyro 

### update config:

def update_config(num_epochs: int, 
            hidden_size: int,
            teacher_enforcing: bool,
            prog_loss_coeff: float,
            a_h_loss_coeff: float,
            latent_loss_coeff: float,
            learning_rate: float,
            episode_length: int,
         ):
    
    TrainConfig.num_epochs = num_epochs
    TrainConfig.hidden_size = hidden_size
    TrainConfig.teacher_enforcing = teacher_enforcing
    TrainConfig.prog_loss_coeff = prog_loss_coeff
    TrainConfig.a_h_loss_coeff = a_h_loss_coeff
    TrainConfig.latent_loss_coeff = latent_loss_coeff
    TrainConfig.learning_rate = learning_rate
    TrainConfig.episode_length = episode_length
     
    cfg_dict = {
        k: v
        for k, v in vars(TrainConfig).items()
        if not k.startswith("__")}

    return cfg_dict



def main(
        num_epochs: int = TrainConfig.num_epochs, 
        hidden_size: int = TrainConfig.hidden_size,
        teacher_enforcing: bool = TrainConfig.teacher_enforcing,
        prog_loss_coeff: float = TrainConfig.prog_loss_coeff,
        a_h_loss_coeff: float = TrainConfig.a_h_loss_coeff,
        latent_loss_coeff: float = TrainConfig.latent_loss_coeff,
        learning_rate: float = TrainConfig.learning_rate,
        episode_length: int = TrainConfig.episode_length,
        save_path: str | None = None,
        ):
    
    hyper_params = update_config(num_epochs, 
        hidden_size,
        teacher_enforcing,
        prog_loss_coeff,
        a_h_loss_coeff,
        latent_loss_coeff,
        learning_rate,
        episode_length)
    
    dsl = DSL.init_default_karel()
    
    run = Run()
    run['hparams'] = hyper_params

    model = load_model(TrainConfig.model_name, dsl)
    logger.info("Model Loaded.")

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl)
    logger.info("Data Loader.")

    trainer = Trainer(model, run, save_path)
    logger.info("Started Training...")
    trainer.train(p_train_dataloader, p_val_dataloader)
    
if __name__ == "__main__":
    tyro.cli(main)



