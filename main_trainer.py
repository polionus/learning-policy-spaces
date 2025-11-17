import os 
import sys
from config import Config
from aim import Run



# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsl import DSL
from vae.models import load_model
from vae.program_dataset import make_dataloaders
from vae.trainer import Trainer


if __name__ == "__main__":

    dsl = DSL.init_default_karel()
    run = Run()

    model = load_model(Config.model_name, dsl)
    print("Model Loaded.")

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl)
    print("Data Loaded.")

    trainer = Trainer(model, run)
    print("Trainer Loader")
    trainer.train(p_train_dataloader, p_val_dataloader)

