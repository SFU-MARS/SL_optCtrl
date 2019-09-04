function [controller] = nonlinear_mpc(N,max_iter,d_min,constr,obstacles)
yalmip('clear')

% Parameters
nx = 4; % Number of states
nu = 2; % Number of inputs
nr = 2; % Number of reference states

% Decision variables
xi0 = sdpvar(nx,1); % Initial position
T = sdpvar(1); % Time interval between states
Q = sdpvar(nr,nr); % Error weighting
R = sdpvar(nu,nu); % Control effort weighting
r = sdpvar(repmat(nr,1,N+1),repmat(1,1,N+1)); % Reference signal
e = sdpvar(repmat(nu,1,N+1),repmat(1,1,N+1)); % Reference tracking error
xi = sdpvar(repmat(nx,1,N+1),repmat(1,1,N+1)); % State
u = sdpvar(repmat(nr,1,N),repmat(1,1,N)); % Input

% Object avoidance constants %%% ADDED
A_o = cell(length(obstacles));
b_o = cell(length(obstacles));
lambda = cell(length(obstacles));
for m = 1:length(obstacles)
    [A_o{m},~,b_o{m}] = generate_rectangle_obstacle_constraint_mats(obstacles(m));
    lambda{m} = sdpvar(repmat(4,1,N+1),repmat(1,1,N+1));
end

% Set up objective function and constraints
constraints = [];

constraints = [constraints, xi{1} == xi0];

objective = 0;
for k = 1:N+1
    objective = objective + e{k}'*Q*e{k};
    constraints = [constraints, ...
        e{k} == xi{k}(1:2) - r{k}];
    constraints = [constraints, ...
       constr.vmin <= xi{k}(3) <= constr.vmax]; % Velocity
   
    % Implement object avoidance
    for m = 1:length(obstacles)
        constraints = [constraints, (A_o{m}*[xi{k}(1);xi{k}(2)] - b_o{m})'*lambda{m}{k} >= d_min];
        constraints = [constraints, ((A_o{m}'*lambda{m}{k})'*A_o{m}'*lambda{m}{k})^1/2 <= 1];
        constraints = [constraints, lambda{m}{k} >= 0];
    end
end

for k = 1:N
    objective = objective + u{k}'*R*u{k};
    % Input constraints
    constraints = [constraints, ...
       constr.omegamin <= u{k}(1) <= constr.omegamax, ... % Angular speed
       constr.amin <= u{k}(2) <= constr.amax]; % Acceleration
    
    % MPC constraints
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
parameters_in = {Q,R,T,[r{:}],xi0};
solutions_out = {[u{:}],[xi{:}]};
ops = sdpsettings('solver','ipopt','ipopt.max_iter',max_iter);
%ops = sdpsettings('solver','fmincon');
controller = optimizer(constraints,objective,ops,parameters_in,...
    solutions_out);
end

