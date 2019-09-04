#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"




# ------- training agent using PPO algorithm -------
~/.conda/envs/light_test/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=None \
--vf_load=True --pol_load=True

~/.conda/envs/light_test/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=None \
--vf_load=True --pol_load=False

~/.conda/envs/light_test/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=None \
--vf_load=False --pol_load=True

~/.conda/envs/light_test/bin/python $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=None \
--vf_load=False --pol_load=False