function [controller] = nonlinear_mpc(N,max_iter,T,Q,R,constr)
%NONLINEAR MPC
% Generate a nonlinear MPC controller using Yalmip to interface with IPOPT
% Inputs
% N: length of horizon
% max_iter: maximum number of iterations in IPOPT
% T: sampling period
% Q: weight matrix for states
% R: weight matrix for input (control effort)
% Outputs
% controller: Yalmip object which takes input parameters, solves for
%   decision variables and returns output parameters

yalmip('clear')

% Parameters
nx = 4; % Number of states
nu = 2; % Number of inputs
np = 2; % Number of position states

% Decision variables
xi0 = sdpvar(nx,1); % Initial state
r = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % Reference signal
e = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % Reference tracking error
xi = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % State
u = sdpvar(repmat(nu,1,N),repmat(1,1,N)); % Input
G = sdpvar(repmat(np,1,N+1), repmat(np,1,N+1), 'full'); % Matrices for 
%   boundary constraint

% Set up objective function and constraints
constraints = [];

% Constrain initial position
constraints = [constraints, xi{1} == xi0];

objective = 0;

% Loop over N+1 states
for k = 1:N+1
    objective = objective + e{k}'*Q*e{k};
    constraints = [constraints, ... 
        e{k} == xi{k} - r{k}]; % Error between states and reference
   
    % State constraints
    constraints = [constraints, ...
    constr.vmin <= xi{k}(3) <= constr.vmax]; % Velocity constraint
    
    % Boundary constraint
    constraints = [constraints, ...
        -1 <= G{k}(1,:)*(xi{k}(1:2) - r{k}(1:2)) <= 1];
end

% Loop over N inputs
for k = 1:N
    objective = objective + u{k}'*R*u{k};
    % Input constraints
    constraints = [constraints, ...
       constr.omegamin <= u{k}(1) <= constr.omegamax, ...
       constr.amin <= u{k}(2) <= constr.amax];
   
    % MPC state constraints
    dotx = xi{k}(3)*cos(xi{k}(4));
    doty = xi{k}(3)*sin(xi{k}(4));
    dotv = u{k}(2);
    dottheta = u{k}(1);
    constraints = [constraints, xi{k+1}(1) == xi{k}(1) + T*dotx];
    constraints = [constraints, xi{k+1}(2) == xi{k}(2) + T*doty];
    constraints = [constraints, xi{k+1}(3) == xi{k}(3) + T*dotv];
    constraints = [constraints, xi{k+1}(4) == xi{k}(4) + T*dottheta];
end

% Set up controller
parameters_in = {[r{:}],[G{:}],xi0};
solutions_out = {[u{:}],[xi{:}]};
ops = sdpsettings('solver','ipopt','ipopt.max_iter',max_iter);
controller = optimizer(constraints,objective,ops,parameters_in,...
    solutions_out);
end

