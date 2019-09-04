% Simulate Trajectory Solver
%   Solve for a trajectory (inputs and states) from a starting state to a
%   goal region for 6D Quadrotor

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

% Starting states for system (circle of radius 0.5 starting at (3.2,3))
start = struct;
start.x = 4.5;
start.vx = 0;
start.z = 3;
start.vz = 0.1;
start.phi = 0;
start.w = 0;

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
opts.d_min = 0.1; % minimum distance between obstacles and quadrotor
opts.smooth_input_on = true; % use R matrix to smooth input as opposed to 
    % minmise input
opts.warmstart = true; % specify whether to apply warmstart or not
controller = mpc_quad_6D_traj_solver(opts,params,weights,goal,env,warmstart);

%% Solve Trajectory

% Initial state
xi0 = [start.x; start.vx; start.z; start.vz; start.phi; start.w];

% Solve for trajectory over length of horizon
tic
[solutions,problem,~,~,P] = controller{xi0};
toc
U = solutions{1}; % Inputs
Xi = solutions{2}; % States

%% Calculate Trajectory with ODE Solver (Open-loop control with solved inputs)

Xi_ode45 = zeros(size(Xi));
tspan = [0 0];
Xi_ode45(:,1) = xi0;
for k = 1:opts.N
    tspan = [tspan(2) tspan(2)+opts.T];
    [t,xi_ode] = ode45(@(t,xi_ode) odefun_quad_6D(t,params,xi_ode,...
        U(:,k)),tspan,Xi_ode45(:,k));
    Xi_ode45(:,k+1) = xi_ode(length(xi_ode),:)';
end

figure(6)
plot(Xi(1,:),Xi(3,:))
hold on
plot(Xi_ode45(1,:),Xi_ode45(3,:))
hold off

%% Calculate next-states using ODE solver setting initial state to be solved inputs
Xi_ode2 = zeros(size(Xi));
Xi_ode2(:,1) = xi0;
tspan = [0 opts.T];
for k = 1:opts.N
    [t,xi_ode] = ode45(@(t,xi_ode) odefun_quad_6D(t,params,xi_ode,...
        U(:,k)),tspan,Xi(:,k));
    Xi_ode2(:,k+1) = xi_ode(length(xi_ode),:)';
end

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

% Positions from trajectory optimizer
scatter(Xi(1,:), Xi(3,:), 'b')
xlabel('x')
ylabel('z')
title('Quadrotor Positions')

% Next-states from ODE solver
scatter(Xi_ode2(1,:), Xi_ode2(3,:), 'g')

% Positions from ODE solver
% scatter(Xi_ode1(1,:), Xi_ode1(3,:), 'r')

plot_obstacle

plot(Xi(1,:), Xi(3,:), 'b')
for k = 1:opts.N
    plot([Xi(1,k) Xi_ode2(1,k+1)],[Xi(3,k) Xi_ode2(3,k+1)],'g')
end
legend('Goal Region','Start Region','Boundary','Trajectory Solver', 'ODE45 Solver Next Step','Obstacles')
hold off

% States
figure(2)
subplot(3,2,1)
plot(Xi(1,:))
ylabel('x')
subplot(3,2,2)
plot(Xi(2,:))
ylabel('vx')
subplot(3,2,3)
plot(Xi(3,:))
ylabel('z')
subplot(3,2,4)
plot(Xi(4,:))
ylabel('vz')
subplot(3,2,5)
plot(Xi(5,:))
ylabel('phi')
subplot(3,2,6)
plot(Xi(6,:))
ylabel('w')
sgtitle('States')

% Inputs
figure(4)
subplot(2,1,1)
plot(U(1,:))
ylabel('T1')
subplot(2,1,2)
plot(U(2,:))
ylabel('T2')
sgtitle('Inputs')

