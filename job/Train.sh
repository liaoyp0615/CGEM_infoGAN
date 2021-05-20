#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=mlgpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=infoGAN
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/2021code/infoGAN/Model/job/train.out
#SBATCH --error=/hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/2021code/infoGAN/Model/job/train.err
 
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30720
#SBATCH --cpus-per-task=2

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
hostname
df -h
cd /hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/2021code/infoGAN/Model
source /hpcfs/bes/mlgpu/liaoyp/env/jobenv.sh
which python

python /hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/2021code/infoGAN/Model/main.py --restore False --D_restore_pkl_path '/hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/2021code/infoGAN/Model/pkl_path/D_test.pkl' --G_restore_pkl_path '/hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/Model/pkl_path/G_test.pkl' --D_pkl_path /hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/Model/pkl_path/D_test.pkl --G_pkl_path /hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/Model/pkl_path/G_test.pkl --datafile /hpcfs/bes/mlgpu/liaoyp/jupyter/CGEM/datasets/xzt_3D_92_10000.root --num_hist 3500 --n_epochs 400 --batch_size 25 --lr 2e-4 --latent_dim 62 --code_dim 2 --n_classes 10 --img_size 92
