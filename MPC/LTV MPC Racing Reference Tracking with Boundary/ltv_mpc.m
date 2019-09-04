function [controller] = ltv_mpc(N,Q,R,constr)
%NONLINEAR MPC
% Generate a LTV MPC controller using Yalmip to interface with IPOPT
% Inputs
% N: length of horizon
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
r = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % Reference signal
e = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % Reference tracking error
xi = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % New state solution
delta_xi = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % State delta variable
xi_prev = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % Previous state guess
u = sdpvar(repmat(nu,1,N),repmat(1,1,N)); % New input solution
delta_u = sdpvar(repmat(nu,1,N),repmat(1,1,N)); % Input delta variable
u_prev = sdpvar(repmat(nu,1,N),repmat(1,1,N)); % Previous input guess
G = sdpvar(repmat(np,1,N+1), repmat(np,1,N+1), 'full'); % Matrices for boundary
% constraint
A = sdpvar(repmat(nx,1,N+1),repmat(nx,1,N+1),'full'); % State matrices
B = sdpvar(repmat(nx,1,N+1),repmat(nu,1,N+1),'full'); % Input matrices

% Set up objective function and constraints
constraints = [];

% Constrain initial state
constraints = [constraints, delta_xi{1} == [0;0;0;0]];

% Set relationship between old states, delta state and current state
% solution

objective = 0;

% Loop over N+1 states
for k = 1:N+1
    objective = objective + e{k}'*Q*e{k};
    constraints = [constraints, ...
        e{k} == xi{k} - r{k}];
   
    % State constraints
    constraints = [constraints, ...
    constr.vmin <= xi{k}(3) <= constr.vmax]; % Velocity
    
    % Boundary constraints
    constraints = [constraints, ...
        -1 <= G{k}(1,:)*(xi{k}(1:2) - r{k}(1:2)) <= 1];
    
    % Linerisation variable constraint
    constraints = [constraints, xi{k} == delta_xi{k} + xi_prev{k}];
end

% Loop over N inputs
for k = 1:N
    objective = objective + u{k}'*R*u{k};
    % Input constraints
    constraints = [constraints, ...
       constr.omegamin <= u{k}(1) <= constr.omegamax, ...
       constr.amin <= u{k}(2) <= constr.amax];
   
    % LTV state space equation constraint
    constraints = [constraints, ...
        delta_xi{k+1} == A{k}*delta_xi{k} + B{k}*delta_u{k}];
    
    % LTV variable constraints
    constraints = [constraints, u{k} == delta_u{k} + u_prev{k}];
end

% Inputs for solver
% u_prev: previous guess for inputs along horizon
% xi_prev: previous guess for states along horizon
% r: state references along horizon
% A: linearised state-transition matrices along horizon
% B: linearised input matrices along horizon
% G: boundary transformation matrices along horizon
parameters_in = {[u_prev{:}],[xi_prev{:}],[r{:}],[A{:}],[B{:}],[G{:}]};

% Outputs for solver
% u: solved inputs along horizon
% xi: solved states along horizon
solutions_out = {[u{:}],[xi{:}]};

% Set up optimizer
% Set to optimize using Gurobi. 'gurobi' can be replaced with another
% solver such as 'quadprog' if desired.
ops = sdpsettings('solver','gurobi');
controller = optimizer(constraints,objective,ops,parameters_in,...
    solutions_out);
end

