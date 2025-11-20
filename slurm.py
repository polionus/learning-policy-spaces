import tempfile
import subprocess
from time import sleep
from termcolor import colored
import hashlib


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

#SBATCH --job_name={job_name} 
#SBATCH --time={submit_time}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mail-user=ashrafi2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=aip-lelis
#SBATCH --gpus={int(gpu_flag)}

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip install --no-index -r compute-canada-reqs.txt

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
    # job_name = f"exp_{_hash[:6]}"
    # log_dir = "./logs"
    # os.makedirs(log_dir, exist_ok=True)

    slurm_script = generate_slurm_script(cmd,
                                        gpu_flag, 
                                        submit_time,
                                        memory,
                                        cpus_per_task,
                                        job_name,
                                        )
    print(slurm_script)
    exit()
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