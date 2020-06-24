#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

# sleep 7200s

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/data_gen.py

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/quick_trainer.py

########### Tuning A2C Hyperparamters ################17-June
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=5 --timesteps_per_actorbatch=512

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=9e-5 --num_ppo_iters=150 --optim_epochs=10 --timesteps_per_actorbatch=1024

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=150 --optim_epochs=10 --timesteps_per_actorbatch=1024

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=9e-6 --num_ppo_iters=150 --optim_epochs=10 --timesteps_per_actorbatch=1024

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=1e-4 --num_ppo_iters=300 --optim_epochs=5 --timesteps_per_actorbatch=512
######################################################

########### Tuning A2C Hyperparamters ################18-June
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=1e-4 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=512

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=1e-5 --num_ppo_iters=400 --optim_epochs=10 --timesteps_per_actorbatch=512 --kl=0.5

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=3e-5 --num_ppo_iters=1000 --optim_epochs=5 --timesteps_per_actorbatch=128 --kl=0.05

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=0.00015 --num_ppo_iters=500 --optim_epochs=5 --timesteps_per_actorbatch=512 --kl=0.15

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=500 --optim_epochs=10 --timesteps_per_actorbatch=512 --kl=0.1

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=2e-5 --num_ppo_iters=150 --optim_epochs=10 --timesteps_per_actorbatch=2048 --kl=0.1

############### /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=2e-5 --num_ppo_iters=500 --optim_epochs=10 --timesteps_per_actorbatch=512 --kl=0.1

########### /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=150 --optim_epochs=10 --timesteps_per_actorbatch=2048 --kl=0.1

######################################################

########### Tuning A2C Hyperparamters ################19-June
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.5

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.1

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=9e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.1

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=2e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.1

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
# --optim_stepsize=2e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.1
######################################################

########### Tuning A2C Hyperparamters ################20-June
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=vi_gd --vf_switch=no --seed=1000 --method=ppo \
# --optim_stepsize=3e-4 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.5

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.15

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
# --optim_stepsize=9e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.15
######################################################

########### Tuning A2C Hyperparamters ################21-June
/media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
--optim_stepsize=3e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.05 --gamma=0.8

/media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
--optim_stepsize=3e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.1 --gamma=0.8

/media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
--vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c \
--optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024 --kl=0.05 --gamma=0.8
######################################################



########### Training the A2C baseline ################
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no --vf_type=boltzmann --vf_switch=always --seed=1000 --method=a2c --optim_stepsize=3e-5

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no  --vf_type=boltzmann --vf_switch=always --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no  --vf_type=boltzmann --vf_switch=always --seed=3000 --method=a2c
######################################################

####### Training the A2C initialization fixed ########
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=6e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=no --seed=3000 --method=a2c
######################################################


############ Training the A2C switch #################
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=1000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=2000 --method=a2c

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=boltzmann --vf_switch=yes --seed=3000 --method=a2c
#######################################################

############ Training the PPO baseline ################
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no  --vf_type=boltzmann --vf_switch=always --seed=1000 --method=ppo

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no  --vf_type=boltzmann --vf_switch=always --seed=2000 --method=ppo

# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=no  --vf_type=boltzmann --vf_switch=always --seed=3000 --method=ppo
#######################################################

############ Training the A2C interpolation ###########
# /media/anjian/Data/anaconda3/envs/py35_Francis/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle \
# --vf_load=yes  --vf_type=vi_gd --vf_switch=no --seed=1000 --method=a2c \
# --optim_stepsize=3e-5 --num_ppo_iters=300 --optim_epochs=10 --timesteps_per_actorbatch=1024

#######################################################



