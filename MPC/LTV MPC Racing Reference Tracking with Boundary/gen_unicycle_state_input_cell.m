function [A,B] = gen_unicycle_state_input_cell(Xi,N,T)
%GEN_STATE_INPUT_MAT Generates state and input matrices for each column of
%the array Xi (each column is new set of states)
% Inputs
% Xi: 4xN array of states along horizon
% N: Length of horizon
% T: Sampling period

A = cell(1,N+1);
B = cell(1,N+1);
% Generate state matrices
for i = 1:N+1
    A{i} = [ 1, 0, T*cos(Xi(4,i)), -T*Xi(3,i)*sin(Xi(4,i));
        0, 1, T*sin(Xi(4,i)), T*Xi(3,i)*cos(Xi(4,i));
        0, 0, 1, 0;
        0, 0, 0, 1];
end

% Generate input matrices
for i = 1:N+1
    B{i} = [-(T^2*Xi(3,i)*sin(Xi(4,i)))/2, (T^2*cos(Xi(4,i)))/2;
        (T^2*Xi(3,i)*cos(Xi(4,i)))/2, (T^2*sin(Xi(4,i)))/2;
        0, T;
        T, 0];
end

end

