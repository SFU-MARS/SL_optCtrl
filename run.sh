#!/usr/bin/env bash

BASEDIR=$(dirname "$0")



# ------- training agent using ddpg algorithm ------ #
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ourddpg.py --policy OurDDPG --env DubinsCarEnv-v0 \
#--initQ=no --save_model




# ------- training agent using PPO algorithm ------- #

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --vf_switch=always

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
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=vi_gd --vf_switch=no




#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/data_gen.py
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/quick_trainer.py

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes



#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes
#

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=mpc --vf_switch=yes



#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no






#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --vf_switch=yes
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --vf_switch=no
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
#--vf_load=no --pol_load=no --vf_switch=yes