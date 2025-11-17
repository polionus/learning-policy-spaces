import polars as pl
from dataclasses import asdict, make_dataclass, field
from typing import List, Any
from index.index import Index
import configparser
import os
from src.utils.hash import hash_list
from src.utils.metadata import get_register_mode
from src.utils.io import IORegistry, write_meta_data_ini, write_meta_data_txt

config_parser = configparser.ConfigParser()


def build_experiment_data_class(config_dict: dict):
    fields = [(k, type(v), field(default=v)) for k, v in config_dict.items()]
    
    fields.append(('run_id', int, field(default=Index.get_run_id() or 0)))

    Experiment = make_dataclass("Experiment", fields)

    def get_meta_data_path(self):
        return os.path.join('index', 'results', 'metadata', f'{self.run_id}.json')

    def get_metadata(self):
        _metadata = asdict(self)
        _metadata.pop('run_id', None)
        return _metadata

    setattr(Experiment, "get_meta_data_path", get_meta_data_path)
    setattr(Experiment, "get_metadata", get_metadata)

    return Experiment

class ExperimentResult:
    def __init__(self, experiment, fields: List[str]):

        
        #fields are the columns of data we would like to save
        self.experiment = experiment
        self.data = {_field:[] for _field in fields}
        self.metadata = experiment.get_metadata()
        self.register_mode = get_register_mode()
        self.file_extension = self.metadata.get('extension', 'csv') ##TODO: Find a way not to repeat this.

        ##Get the function that writes the data
        self.data_write_fn = IORegistry.get_write_fn(self.file_extension)

        self.exp_config_str, self.exp_hash = self.get_config_and_hash()
        ## Initialize the directory if not already there.
        self.init_dirs()
        

    def init_dirs(self):
        self.metadata_dir_path = os.path.join(Index.results_path,'metadata')
        self.results_dir_path = os.path.join(Index.results_path, 'data')
        os.makedirs(self.results_dir_path, exist_ok=True)
        os.makedirs(self.metadata_dir_path, exist_ok=True)

        if not self.register_mode:
            self.temp_path = os.path.join('index', 'temp', self.exp_hash)
            ##Then make a directory with the hash as the name: 
            os.makedirs(self.temp_path, exist_ok=True)


    def get_experiment(self):
        return self.experiment

    def get_metadata(self):
        return self.metadata

    def get_config_and_hash(self):
        config_str = str(self.get_metadata())
        hash = hash_list(self.get_metadata())
        return config_str, hash
    
    def collect(self, field_vals: dict[Any]):
        for _field, value in field_vals.items():
            self.data[_field].append(value)
    
    def save(self):


        if self.register_mode:
            result_path = os.path.join(self.results_dir_path, f'{self.experiment.run_id}.{self.file_extension}')
            metadata_path = os.path.join(self.metadata_dir_path, f'{self.experiment.run_id}.ini')
            write_meta_data_ini(self.metadata, metadata_path)

            Index.add_experiment(config_str=self.exp_config_str, hash=self.exp_hash) 

        else:
            result_path = os.path.join(self.temp_path, f'run.{self.file_extension}')
            metadata_path = os.path.join(self.temp_path, 'run.txt')
            write_meta_data_txt(self.metadata, metadata_path)

        self.data_write_fn(self.data, result_path)
        
            

        



    