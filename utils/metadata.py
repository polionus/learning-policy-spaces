from src.utils.hash import hash_list
from index.index import Index
from typing import List, Tuple
import json 
import os


def get_meta_data():
    return json.loads(os.environ['metadata'])

def get_register_mode():
    return json.loads(os.environ['register'])




### If I am changing the metadata, I need to chagne everything about this file!
class MetaDataManager:

    def __init__(self):
        #A dictionary to maintain the runs and their seeds
        self.experiment_desc = {}

    def get_run_config_tuple(self, metadata: List):
        metadata.pop('run_id', None)
        run_config = metadata.copy()
        run_config.pop('search_seed', None)

        return tuple(run_config.items())
    
    def get_seed(self, metadata):
        return metadata.get('search_seed', 0)

    def insert_experiment_and_get_hash(self, metadata: dict):

        run_config = self.get_run_config_tuple(metadata)
        seed = self.get_seed(metadata)

        if run_config not in self.experiment_desc:
            self.experiment_desc[run_config] = {}

        ##I am getting the right number of hashes, so this is not the problem.
        _hash = hash_list(metadata)
        self.experiment_desc[run_config][seed] = _hash

        return _hash

    def get_experiment_desc(self):
        return self.experiment_desc
    

    def get_experiment_hash_list(self, run_config: Tuple[str, str, int]):
        return list(self.experiment_desc[run_config].values())
    
    def get_experiment_ids(self, run_config: Tuple[str, str, int]):
        return Index.get_id_list_by_hash(self.get_experiment_hash_list(run_config))

