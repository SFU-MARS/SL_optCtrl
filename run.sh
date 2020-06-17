#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/data_gen.py

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/quick_trainer.py


########### Training the A2C baseline ################
/media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=1000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=3000 --method=a2c
######################################################

####### Training the A2C initialization fixed ########
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=no --seed=3000 --method=a2c
######################################################


############ Training the A2C switch #################
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes --pol_load=no --vf_type=boltzmann --vf_switch=yes --seed=3000 --method=a2c
#######################################################

############ Training the PPO baseline ################
/media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=1000 --method=ppo

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=2000 --method=ppo

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --pol_load=no --vf_type=boltzmann --vf_switch=always --seed=3000 --method=ppo
#######################################################
