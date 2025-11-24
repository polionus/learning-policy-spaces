import tempfile
import subprocess
from time import sleep
from termcolor import colored
import hashlib



ARCHIVE_NAME = "$SLURM_TMPDIR/learning_policy_spaces_env.tar.gz"
VENV_DIR = "$SLURM_TMPDIR/cached_env"

def get_hash_name(cmd: str, length: int = 7):
    return f"exp_{hashlib.sha256(cmd.encode()).hexdigest()[:length]}"

def generate_slurm_script(cmd, 
                          gpu_flag: bool, 
                          submit_time: str,
                          memory: str,
                          cpus_per_task: int,
                          job_name: str,
                          ):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out 
#SBATCH --time={submit_time}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mail-user=ashrafi2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=aip-lelis
#SBATCH --gpus={int(gpu_flag)}

module --force purge
module load StdEnv/2023
module load python/3.12 rust cuda/12.2 swig clang

export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/env

uv venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
uv sync --extra jax-cuda


#Important magic to make neural networks fast. Will not work with multi-threading
export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 


{cmd}
"""

def submit(cmd, 
        gpu_flag: bool, 
        submit_time: str,
        memory: str,
        cpus_per_task: int,
        job_name: str
        ):

    slurm_script = generate_slurm_script(cmd,
                                        gpu_flag, 
                                        submit_time,
                                        memory,
                                        cpus_per_task,
                                        job_name,
                                        )
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".slurm") as f:
        f.write(slurm_script)
        slurm_path = f.name
    
    sleep(0.001)
    subprocess.run(['sbatch', slurm_path])


def main():

    cmd = input(colored("Please input your command: ", 'red'))
    gpu_flag = input(colored("GPU?: ", 'blue'))
    submit_time = input(colored('Submit Time: ', 'green'))
    memory = input(colored("Memory: ", 'red'))
    cpus_per_task = input(colored("CPUs Per Task: ", 'blue'))

    job_name = get_hash_name(cmd)
    
    submit(cmd,
            gpu_flag, 
            submit_time,
            memory,
            cpus_per_task,
            job_name
            )
    
if __name__ == "__main__":
    main()