#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"




# ------- training agent using PPO algorithm -------
python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle
