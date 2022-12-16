#!/bin/bash
# uruchamiamy bezposrednio z konsoli, przykladowo za pomoca: sbatch -n1 -N1-1 --job-name=cubT3 --qos=big --gres=gpu:1 --mem=32G --partition=student --cpus-per-task=16 CUB_test3.sh
# albo piszesz kolejne instrukcje typu
#SBATCH --job-name=multimaml
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=student
#SBATCH --cpus-per-task=8


# (musi byc # na poczatku przed SBATCH) obczaj sobie instrukcje, jak dziala slurm https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html

# TU TWOJE TOKENY Z NEPTUNA
export NEPTUNE_PROJECT=arturgorak/MAML
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NTdjYThjZS04MGY1LTQzOGUtYWU2Ni1hOTIwOGJlNmFmNjcifQ==

# TU ZMIEN NA TWOJE SRODOWISKO W CONDZIE !!!
source activate my_env


python train.py --method maml --model Conv4 --dataset cross_char --num_classes 4112 --n_shot 1 --test_n_way 5 --train_n_way 5 --stop_epoch 64 --lr 1e-2 --maml_adapt_classifier

