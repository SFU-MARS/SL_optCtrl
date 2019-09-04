%% Reference Tracking Simulation
% Using LTV MPC approach for controlling system with sparse QP matrix.

clc
clear all
rng(0)
%% Set Starting State and Point that the vehicle will track
start_x = 0;
start_y = 0;
start_v = 0;
start_theta = 0;

end_x = 1;
end_y = 1.03;

% System constraints

constr.amax = 0.25;
constr.amin = -constr.amax;
constr.omegamax = 2;
constr.omegamin = -constr.omegamax;
constr.vmax = 0.6;
constr.vmin = -constr.vmax;

%% Set Obstacle
% Set location, width and height of obstacles
obs1 = struct;
obs1.center_x = 0.8;
obs1.center_y = 0.7;
obs1.width = 0.5;
obs1.height = 0.5;
[~,bound1,~] = generate_rectangle_obstacle_constraint_mats(obs1);

obs2 = struct;
obs2.center_x = 0.2;
obs2.center_y = 0.2;
obs2.width = 0.2;
obs2.height = 0.2;
[~,bound2,~] = generate_rectangle_obstacle_constraint_mats(obs2);

obstacles = [obs1 obs2];
boundaries = [bound1' bound2'];

d_min = 0.03; % minimum distance between obstacles and vehicle (point)

%% Load Controller
% Load controller for MPC
N = 30;
max_iter = 20;
controller = nonlinear_mpc_static_obstacle(N, max_iter,d_min,constr,obstacles);

%% Simulation
% Here we are running through the time steps to simulate the system using
% MPC for control
start_time = 0;
end_time = 20;
T = 0.05; % Sampling period
times = start_time:T:end_time;

% Constant terms
N_STATES = 4;
N_INPUTS = 2;

% Weight matrices
R = diag([0.01 0.01]); % Penalise inputs
Q = diag([5 5]);

% Set references
ref = [end_x*ones(1,length(times)+N); end_y*ones(1,length(times)+N)];

% Initial states/inputs
init_omega = 0;
init_a = 0;
init_v = start_v;
init_x = start_x;
init_y = start_y;
init_theta = start_theta;

% Initialise inputs
u = [init_omega, init_a]';  % Initial input
U = repmat(u, 1, N); % Guess for inputs across horizon

% Initialise states
xi = [init_x,init_y,init_v, ...
    init_theta]'; % Initial states
Xi = zeros(length(xi), N+1); % Guess for states across horizon

% History of implemented inputs and states
history_U = [];
history_Xi = xi;

% Loop through times
tspan = [0 0];
fprintf('Simulation Progress:         ')
for i = 1:length(times)-1 % starting off at first time already (not zeroth)
    fprintf('\b\b\b\b\b\b\b\b%6.2f %%',(i/(length(times)-1)*100))

    % Optimise for input via MPC
    solutions = controller{Q,R,T,ref(:,i:i+N),xi};
    U = solutions{1};
    Xi = solutions{2};
            
    % Apply input to plant
    tspan = [tspan(2) tspan(2)+T];
    [t,x_ode] = ode45(@(t,x_ode) ode_unicycle_fun_a_input(t,x_ode,U(:,1)), tspan, xi);
    xi = x_ode(length(x_ode),:)';
    %xi = Xi(:,1);
    % Update history
    history_U = [history_U,U(:,1)];
    history_Xi = [history_Xi,xi];
    
    % Plot reference trajectory and state
    plot_moving_traj
    plot_obstacle
    plot_prediction

    % Repare guess of inputs and states along horizon for next iteration
    U = [U(:,2:N), zeros(2,1)]; % keep last one as zero for now
    Xi = [xi, zeros(length(xi), N)];
end
fprintf('\n')
figure(1)
legend('Reference Trajectory','Simulated Path')

%% Plot States and Inputs

plot_states
plot_inputs
