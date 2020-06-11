#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no

# /local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v1 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --pol_load=no --vf_switch=always


# ------- training agent using ddpg algorithm ------ #
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ourddpg.py --policy TD3 --env PlanarQuadEnv-v0 \
#--save_model --initQ no

# /local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ourddpg.py --policy TD3 --env PlanarQuadEnv-v0 \
# --save_model --initQ no  --useGD  --save_debug_info

# /local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ourddpg.py --policy TD3 --env PlanarQuadEnv-v0 \
# --save_model --initQ no  --useValInterp  --save_debug_info



# ------- training agent using PPO algorithm ------- #

# /local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no

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