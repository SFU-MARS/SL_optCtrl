#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"

# ------- training agent using TRPO algorithm ------ #
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_trpo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --set_additional_goal=angle \
#--vf_load=no --pol_load=no



# ------- training agent using PPO algorithm ------- #

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no


#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --adv_shift=yes

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --adv_shift=yes

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --adv_shift=yes

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no






