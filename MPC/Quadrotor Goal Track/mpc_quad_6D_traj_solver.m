function [solver] = mpc_quad_6D_traj_solver(opts,params,weights,goal,env,warmstart)
%MPC_TRAJECTORY_PLANNEER
%   Generates Yalmip solver object for mpc-based trajectory solving
%   INPUTS
%   opts: struct with opts.opts.N (horizon length), opts.opts.max_iter 
%       (maximum number of iterations for IPOPT solver)
%   params: model parameters for quadrotor system (object loaded from
%       'get_quad_6D_params.m' file
%   weights: struct with weights.weights.Q (position error weighting) and 
%       weights.weights.R (control effort weighting)
%   goal: struct with goal.xmin, goal.xmax, goal.ymin, goal.ymax (box
%       constrains for goal region) as well as goal.vxmin/goal.vymin and
%       goal.vxmax/goal.vymax (velocity constraints in goal region)
%   env: struct whose fields consists of list 'obst' and struct 'bounds'
%       env.obst: list of obstacle structs with fields center_x, center_y,
%           width and height
%       env.bounds: struct consisting of xmin, xmax, ymin and ymax fields
%           indicating the boundaries of the environment
%   OUTPUTS
%   solver: Yalmip solver which can be called with 
%       'solutions = solver{xi0}' where xi0 is a 6x1 array filled with the
%       inital states

yalmip('clear')

% Parameters
nxi = 6; % Number of states
nu = 2; % Number of inputs

% Decision variables
xi0 = sdpvar(nxi,1); % Initial state
xi = sdpvar(repmat(nxi,1,opts.N+1),repmat(1,1,opts.N+1)); % State
u = sdpvar(repmat(nu,1,opts.N),repmat(1,1,opts.N)); % Input

% Obstacle avoidance constants/decision variables 
A_o = cell(length(env.obst)); % Cell of A_o matrices
b_o = cell(length(env.obst)); % Cell of b_o matrices
lambda = cell(length(env.obst)); % Cell of lambda deicision variables
for m = 1:length(env.obst)
    [A_o{m},~,b_o{m}] = generate_rectangle_obstacle_constraint_mats(env.obst(m));
    lambda{m} = sdpvar(repmat(4,1,opts.N+1),repmat(1,1,opts.N+1));
end

% Set up objective 
objective = 0;

% Initial state constraint
constraints = [xi{1} == xi0;];

% Final state constraints
if opts.goal_on
    for k = opts.N+1 - opts.goal_steps + 1:opts.N+1
        constraints = [constraints,...
            goal.xmin <= xi{k}(1) <= goal.xmax;
            goal.vxmin <= xi{k}(2) <= goal.vxmax;
            goal.zmin <= xi{k}(3) <= goal.zmax;
            goal.vzmin <= xi{k}(4) <= goal.vzmax;];
    end
end

% Set reference in objective function to be center of goal region
end_center = [(goal.xmin + goal.xmax)/2; (goal.zmin + goal.zmax)/2];
r = [end_center(1); 0; end_center(2); 0; 0; 0];

% Loop through states over horizon
for k = 1:opts.N+1
    % Consider current position in objective function
    if (k > opts.N-5)
        % Weight reference error with S matrix in last 'goal_steps' steps
        objective = objective + (xi{k} - r)' * weights.S * (xi{k} - r);
    else
        % Weight reference error with Q matrix before last 'goal_steps'
        % steps
        objective = objective + (xi{k} - r)' * weights.Q * (xi{k} - r);
    end
    
    
    % Environment boundary constraints
    constraints = [constraints,
        env.bounds.xmin <= xi{k}(1) <= env.bounds.xmax;
        env.bounds.zmin <= xi{k}(3) <= env.bounds.zmax];
    
    % Obstacle avoidance constraints
    for m = 1:length(env.obst)
        constraints = [constraints, ...
            (A_o{m}*[xi{k}(1);xi{k}(3)] - b_o{m})'*lambda{m}{k} >= opts.d_min;
            ((A_o{m}'*lambda{m}{k})'*A_o{m}'*lambda{m}{k})^1/2 <= 1;
            lambda{m}{k} >= 0];
    end
end

% Weight input in objective function
if opts.smooth_input_on
    for k = 1:opts.N-1
        objective = objective + (u{k+1}-u{k})'*weights.R*(u{k+1}-u{k});
    end
else
    for k = 1:opts.N
        objective = objective + u{k}'*weights.R*u{k};
    end
end


% Loop through inputs over horizon
for k = 1:opts.N  
    % Rate of change of states
    dxi(1) = xi{k}(2);
    dxi(2) = (-(1/params.m)*params.transDrag*xi{k}(2))+...
        ((-1/params.m)*sin(xi{k}(5))*u{k}(1))+...
        ((-1/params.m)*sin(xi{k}(5))*u{k}(2));
    dxi(3) = xi{k}(4);
    dxi(4) = (-1/params.m)*(params.m*params.grav +...
        params.transDrag*xi{k}(4)) +...
        ((1/params.m)*cos(xi{k}(5))*u{k}(1))+...
        ((1/params.m)*cos(xi{k}(5))*u{k}(2));
    dxi(5) = xi{k}(6);
    dxi(6) = ((-1/params.Iyy)*params.rotDrag*xi{k}(6))+...
        ((-params.l/params.Iyy)*u{k}(1))+...
        ((params.l/params.Iyy)*u{k}(2));
    
    % Constrain system dynamics
    constraints = [constraints, xi{k+1}(1) == xi{k}(1) + opts.T*dxi(1)];
    constraints = [constraints, xi{k+1}(2) == xi{k}(2) + opts.T*dxi(2)];
    constraints = [constraints, xi{k+1}(3) == xi{k}(3) + opts.T*dxi(3)];
    constraints = [constraints, xi{k+1}(4) == xi{k}(4) + opts.T*dxi(4)];
    constraints = [constraints, xi{k+1}(5) == xi{k}(5) + opts.T*dxi(5)];
    constraints = [constraints, xi{k+1}(6) == xi{k}(6) + opts.T*dxi(6)];
    constraints = [constraints, ...
        params.T1Min <= u{k}(1) <= params.T1Max, ...
        params.T2Min <= u{k}(2) <= params.T2Max];
end

% Check for and apply warm start
usex0 = 0; % flag for warm start in Yalmip set off
if opts.warmstart
    usex0 = 1; % flag for warm start in Yalmip set one
    % Warm start inputs
    for i = 1:opts.N
        assign(u{i}, warmstart.U(:,i));
    end
    
    % Warm start states
%     for i = 1:opts.N+1
%         assign(xi{i}, warmstart.Xi(:,i));
%     end
end

% Set up controller
parameters_in = {xi0};
solutions_out = {[u{:}],[xi{:}]};
if opts.max_iter_on
    ops = sdpsettings('verbose',1,'showprogress',1,'solver','ipopt','ipopt.max_iter',opts.max_iter,'usex0',usex0);
else
    ops = sdpsettings('verbose',1,'showprogress',1,'solver','ipopt','usex0',usex0);
end
solver = optimizer(constraints,objective,ops,parameters_in,...
    solutions_out);

end

