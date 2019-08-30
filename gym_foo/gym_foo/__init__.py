import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register envs
# ----------------------------------------
# DubinsCar.worlds
register(
    # For gym id, the correct form would be xxx-v0, not xxx_v0
    id='DubinsCarEnv-v0',
    entry_point='gym_foo.gym_foo.envs:DubinsCarEnv_v0',
)

# PlanarQuadrotor.worlds
register(
    id='PlanarQuadEnv-v0',
    entry_point='gym_foo.gym_foo.envs:PlanarQuadEnv_v0',
    # More arguments here
)

register(
    id='DubinsCarEnv_dqn-v0',
    entry_point='gym_foo.gym_foo.envs:DubinsCarEnv_v0_dqn',
)

register(
    id='PlanarQuadEnv_dqn-v0',
    entry_point='gym_foo.gym_foo.envs:PlanarQuadEnv_v0_dqn',
)


