import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register envs
# ----------------------------------------
# Ackermann
register(
    # For gym id, the correct form would be xxx-v0, not xxx_v0
    id='AckermannEnv-v0',
    entry_point='gym_foo.gym_foo.envs:AckermannEnv_v0',
)

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



