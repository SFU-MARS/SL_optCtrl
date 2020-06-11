A few steps as reminder:

(1) Put raw MPC data (latest for DDPG quad env) here.
(2) Directly learn a 6D valNN model using this raw MPC data. The NN structure for this valNN is not important. Use [64, 64] hidden layers would be fine.
(3) Use Qfunc.py to collect full states in Gazebo and train 16D QNN. The valNN from step (2) has the same role as valM of dubins Car example (used for interpolation). 