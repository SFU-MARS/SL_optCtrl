This qnn_interp is trained by the following:
(1) Collect raw MPC trajectories and convert each state to its' reward, value and qvalue correspondingly.
(2) Then we have (s,a) -> (q_value) pairs as training data.
(3) Supervised Learning-based training 

Potential problem: the control "a" here is from MPC, but may not be suitable to RL environment. (Same problem as policy initialization)