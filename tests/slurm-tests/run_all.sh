#!/bin/bash

CLUSTER=$1

CURRENT_DATE=$(date +%Y-%m-%d)

python tests/slurm-tests/super_49b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/super_49b_evals --expname_prefix $CURRENT_DATE &
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-aligner --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-aligner --wandb_project nemo-skills-slurm-ci --expname_prefix $CURRENT_DATE-omr-simple-recipe-nemo-aligner &
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-rl --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-rl --wandb_project nemo-skills-slurm-ci --expname_prefix $CURRENT_DATE-omr-simple-recipe-nemo-rl &
wait
