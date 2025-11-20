import tempfile
import subprocess
from time import sleep
from termcolor import colored
import hashlib
import tyro



ARCHIVE_NAME = "learning_policy_spaces_env.tar.gz"
VENV_DIR = "cached_env"

def get_hash_name(cmd: str, length: int = 7):
    return f"exp_{hashlib.sha256(cmd.encode()).hexdigest()[:length]}"


def generate_setup_script():
    return f"""#!/bin/bash
rm -rf {ARCHIVE_NAME}
rm -rf {VENV_DIR}

module load python/3.10

# Create the env
echo "Creating uv enviornment..."
uv venv {VENV_DIR}


#Activate the env
source {VENV_DIR}/bin/activate



#Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Deactivate
deactivate


# Compress the enviornment
echo "Compressing the environment ..."
tar -czf {ARCHIVE_NAME} {VENV_DIR}

# Clean up source directory
rm -rf {VENV_DIR}

echo "Setup Successful!"

"""

def setup_env(setup_script: str):

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".sh") as f:
        f.write(setup_script)
        setup_path = f.name

    sleep(0.001)
    activate_cmd = f"chmod +x {setup_path}".split()
    run_command = [setup_path]

    subprocess.run(activate_cmd, check = True)
    subprocess.run(run_command, check = True)


def generate_slurm_script(cmd, 
                          gpu_flag: bool, 
                          submit_time: str,
                          memory: str,
                          cpus_per_task: int,
                          job_name: str,
                          ):
    return f"""#!/bin/bash

#SBATCH --job-name={job_name} 
#SBATCH --time={submit_time}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mail-user=ashrafi2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=aip-lelis
#SBATCH --gpus={int(gpu_flag)}

module load python/3.10
echo "Copying cached environment '{ARCHIVE_NAME}' to fast storage ($SLURM_TMPDIR)..."
cp -f $SLURM_SUBMIT_DIR/{ARCHIVE_NAME} $SLURM_TMPDIR/

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


def main(setup: bool = False):

    if setup:
        setup_script = generate_setup_script()
        setup_env(setup_script)
        return

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
    tyro.cli(main)