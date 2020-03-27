#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"

# ------- training agent using TRPO algorithm ------ #
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_trpo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --set_additional_goal=angle \
#--vf_load=no --pol_load=no



# ------- training agent using PPO algorithm ------- #

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no





#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/data_gen.py

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/quick_trainer.py

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=0.95 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=0.95 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=0.95 --grad_norm=10.0

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=0.5

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=0.5

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --lam=1.0 --grad_norm=0.5


#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/gym_foo/gym_foo/envs/PlanarQuadEnv_v0.py
