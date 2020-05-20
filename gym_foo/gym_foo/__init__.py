import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


# PlanarQuadrotor
register(
    id='PlanarQuadEnv-v0',
    entry_point='gym_foo.gym_foo.envs:PlanarQuadEnv_v0',
    # More arguments here
)

# DubinsCar
register(
    id='DubinsCarEnv-v0',
    entry_point='gym_foo.gym_foo.envs:DubinsCarEnv_v0',
    # More arguments here
)



