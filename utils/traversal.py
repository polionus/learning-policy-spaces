import subprocess
from src.utils.registry import get_script
from index.index import Index
from src.utils.metadata import MetaDataManager
from src.utils.parse_config import parse_config
from src.utils.slurm import submit
import os
import json


def traverse_experiments(config, 
                         register_mode: bool, 
                         submit_time: str = 0, 
                         run: bool = False, 
                         submit_flag: bool = False,
                         gpu_flag: bool = False):

    script_name = get_script(config["script"])
    config.pop('script')

    cmds, args_list = parse_config(config, script_name)

    manager = MetaDataManager()

    for cmd, args in zip(cmds, args_list):

        ### Doing this to avoid the unnecessary gathering of information during an actual run. There is probably a cleaner way, but this is good enough
        metadata = args
        _hash = manager.insert_experiment_and_get_hash(metadata)

        if run:
            if Index.checkhashmap(_hash):
                print("Skipping already-run experiment.")
            else:
                ### Set the current config meta data:
                env = os.environ.copy()
                env['metadata'] = json.dumps(metadata)
                env['register'] = json.dumps(register_mode)


                if submit_flag:
                    submit(_hash, 
                           cmd, 
                           metadata, 
                           register_mode, 
                           gpu_flag,
                           submit_time,
                           )
                else:    
                    subprocess.run(cmd, env= env)
    return manager
