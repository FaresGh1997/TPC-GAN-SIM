#! /bin/bash
#SBATCH --job-name="GPU-Training"
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --output="slurm%j.out"%j.out
#SBATCH --error="slurm%j.err"%j.out

source tf-gpu/bin/activate
module load Python/Tensorflow_GPU_v2.6  CUDA/11.2

nvidia-smi
python3 /home/fgazzavi/My_Project_gpu/Model_8X16_paper.py