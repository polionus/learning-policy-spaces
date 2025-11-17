import os
import json
import tempfile
import subprocess
from time import sleep

def generate_slurm_script(cmd, metadata, register_mode, job_name, log_dir, gpu_flag: bool, submit_time: str):

    cmd = " ".join(cmd)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --time={submit_time}
#SBATCH --mem=12G
#SBATCH --cpus-per-task=32
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

export metadata='{json.dumps(metadata)}'
export register='{json.dumps(register_mode)}'

{cmd}
"""

def submit(_hash: str, cmd, metadata, register_mode, gpu_flag: bool, submit_time: str):
    job_name = f"exp_{_hash[:6]}"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    slurm_script = generate_slurm_script(cmd, metadata, register_mode, job_name, log_dir, gpu_flag, submit_time)

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".slurm") as f:
        f.write(slurm_script)
        slurm_path = f.name
    
    sleep(0.001)
    subprocess.run(['sbatch', slurm_path])