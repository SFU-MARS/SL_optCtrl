import numpy as np

data = np.load("/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/trial_3/value_matrix_9.npy")
print(data.shape) # x-11, z-11, vx-9, vz-9, theta-11, omega-11
new_data = np.zeros([11, 9, 11, 9, 11, 11], dtype = float)

for i, v in np.ndenumerate(data):
    new_index = tuple([i[0], i[2], i[1], i[3], i[4], i[5]])
    new_data[new_index] = v

np.save("/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/trial_3/transferred_value_matrix_9.npy", new_data)

check_list = [tuple([1,2,3,4,3,2]),
              tuple([2,1,4,5,2,2]),
              tuple([3,1,4,2,4,5])]


check_list_2 = [tuple([1,3,2,4,3,2]),
                tuple([2,4,1,5,2,2]),
                tuple([3,4,1,2,4,5])]


for i in range(0, 3):
    print(data[check_list[i]])
    print(new_data[check_list_2[i]])


for i in range(0, 3):
    print(data[check_list_2[i]])
    print(new_data[check_list[i]])