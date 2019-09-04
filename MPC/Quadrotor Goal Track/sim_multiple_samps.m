% Simulate Multiple Samples
%   Solve for a trajectory (inputs and states) for a variety of different
%   initial state samples

%% Setup Problem

% Goal requirements for system
goal = struct;
goal.xmin = 3;
goal.xmax = 5;
goal.zmin = 8;
goal.zmax = 10;
goal.vxmin = -1;
goal.vxmax = 1;
goal.vzmin = goal.vxmin;
goal.vzmax = goal.vxmax;

% Starting circle
start_region = struct;
start_region.x = 3.2;
start_region.z = 3;
start_region.r = 0.5;


%% Set Envirionment
env = struct;

% Obstacles
obs1 = struct;
obs1.center_x = -2;
obs1.center_z = 7;
obs1.width = 1.8;
obs1.height = 1.8;
[~,bound1,~] = generate_rectangle_obstacle_constraint_mats(obs1);

obs2 = struct;
obs2.center_x = -1.5;
obs2.center_z = 3;
obs2.width = 1.5;
obs2.height = 1.5;
[~,bound2,~] = generate_rectangle_obstacle_constraint_mats(obs2);

obs3 = struct;
obs3.center_x = 3;
obs3.center_z = 6;
obs3.width = 1.5;
obs3.height = 1.5;
[~,bound3,~] = generate_rectangle_obstacle_constraint_mats(obs3);

% env.obst = [obs1 obs2]; % Group all obstacles inside a list
env.obst = [obs1 obs2 obs3]; % Group all obstacles inside a list
% env.obst = [];
% boundaries = [bound1' bound2']; % boundary for obstacles (for plotting)
boundaries = [bound1' bound2' bound3'];
% boundaries = [];

% Boundaries for environment
env.bounds.xmin = -5;
env.bounds.xmax = 5;
env.bounds.zmin = 0;
env.bounds.zmax = 10;


%% Setup Solver

% Load Quadrotor parameters
params = get_quad_6D_params;

% Load warm start data
load('warmstart_no_obst_center_circle.mat');

% Set MPC weight matrices
weights.Q = 10*diag([1 0 1 0 1 1]); % Weights for tracking center of endpoint,
    % minimising velocity, minimising pitch angle and pitch angular speed
    % before last 'goal_steps' steps in horizon ([x vx y vy phi w])
weights.R = diag([0.01 0.01]); % Weights for minimising thrust (or smoothing)
weights.S = 10*diag([1 1 1 1 1 1]); % Weights for tracking center of endpoint,
    % minimising velocity, minimising pitch angle and pitch angular speed
    % in the last 'goal_steps' steps ([x vx y vy phi w])

% Load controller for MPC
opts = struct;
opts.T = 0.05; % sampling period
opts.N = 60; % horizon length
opts.max_iter_on = true; % tell controller whether to enforce max iter
opts.max_iter = 200; % maximum iterations for IPOPT solver
opts.goal_steps = 10; % number of steps at end of horizon to enforce goal requirements
opts.goal_on = true; % tell controller whether ot enforce goal constraints
opts.d_min = 0.3; % minimum distance between obstacles and quadrotor
opts.smooth_input_on = true; % use R matrix to smooth input as opposed to 
    % minmise input
opts.warmstart = true; % specify whether to apply warmstart or not
controller = mpc_quad_6D_traj_solver(opts,params,weights,goal,env,warmstart);

%% Solve Trajectory

% load samples
load('init_states_200.mat');

nsamps = 200;
start = struct;
test = struct;
test.times = [];
test.Xi = cell(1,nsamps);
test.U = cell(1,nsamps);
for i = 1:nsamps
    i
    % Initial states
    start.x = init_states.x(i);
    start.vx = init_states.vx(i);
    start.z = init_states.z(i);
    start.vz = init_states.vz(i);
    start.phi = init_states.phi(i);
    start.w = init_states.w(i);
    xi0 = [start.x; start.vx; start.z; start.vz; start.phi; start.w];
    
    % Solve for solution and time
    tic
    [solutions,problem,~,~,P] = controller{xi0};
    solve_time = toc;
    test.times = [test.times toc];
    test.U{i} = solutions{1}; % Inputs
    test.Xi{i} = solutions{2}; % States
end



% Save data
save('test_samps.mat','test')

%% Plot Trajectory

% Draw obstacles and goal region
figure(1)
draw_goal(goal)
hold on
grid on

% Draw starting region
circle(start_region.x, start_region.z, start_region.r, 'm')

% Plot boundaries for of environment
plot([env.bounds.xmin,env.bounds.xmin, ...
    env.bounds.xmax, env.bounds.xmax, env.bounds.xmin], ...
    [env.bounds.zmin,env.bounds.zmax, ...
    env.bounds.zmax, env.bounds.zmin, env.bounds.zmin],'k')

% Plot obstacles
plot_obstacle


% Plot positions for samples
nanlist = [];
for i = 1:nsamps
    
    % check for nans in data
    if (sum(isnan(test.U{i})) > 0)
        nanlist = [nanlist, true];
        continue
    else
        nanlist = [nanlist, false];
    end
    plot(test.Xi{i}(1,:),test.Xi{i}(3,:),'b');
end



