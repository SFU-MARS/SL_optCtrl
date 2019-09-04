%% Reference Tracking Race Track
% Using LTV MPC to track center line of race track with boundary
% constraints. Simulation exits when after the 0th reference along the
% horizon cycles back to the first coordinate for the center of the track.
% The guesses for inputs is initialised with nonlinear MPC.
%
% Track ('track2.mat') sourced from https://github.com/alexliniger/MPCC
%
% Current nonlinear solver is IPOPT. This can be found on its website
% https://projects.coin-or.org/Ipopt or alternatively packaged with the 
% OPTI toolbox https://inverseproblem.co.nz/OPTI/
%
% Default qp solver for ltv is GUROBI (https://www.gurobi.com/).
% Another solver can be manually specified within the ltv_mpc function.

% Unicycle model state variables
% States
% x - position along x-axis
% y - position along y-axis
% v - velocity
% theta - heading of unicycle

% Inputs
% omega - angular velocity of unicycle
% a - acceleration of unicycle

clc
clear all

%% Settings

% Indicate whether to plot vehicle at each time step in the loop. Slows 
% down simulation, but allows debugging of weird behaviour.
plot_vehicle_each_step = true;

N = 30; % Length of horizon
M = 5; % Number of iterations to repeat LTV for single time
nlinear_max_iter = 10; % Max iterations for IPOPT solver
T = 0.05; % Sampling period

% Specify system constraints
% Note that x-y position constraints cannot be set here since they are
% constrained by the track boundaries in the controller
constr = struct;
constr.amax = 1; % Max acceleration
constr.amin = -constr.amax; % Min acceleration
constr.omegamax = 2; % Max angular velocity
constr.omegamin = -constr.omegamax; % Min angular velocity
constr.vmax = 1; % Max velocity
constr.vmin = 0; % Min velocity
% Specify Weights
R = diag([0.3 0.3]); % [omega weight, a weight]
Q = diag([10 10 0 0]); % [x weight, y weight, v weight, theta weight]

%% Load Controller

% Nonlinear controller for initialisation 
controller_init = nonlinear_mpc(N, nlinear_max_iter,T,Q,R,constr);

% LTV controller
controller_sim = ltv_mpc(N,Q,R,constr);

%% Simulation

% Constant terms
N_STATES = 4; % States in unicycle model
N_INPUTS = 2; % Inputs in unicycle model

% Initial states/inputs
init_omega = 0;
init_a = 0;
init_v = 0;

% Set center of track as reference signal
load('track2.mat');
x_path = track2.center(1,:);
y_path = track2.center(2,:);
x_ref = [x_path, x_path];
y_ref = [y_path, y_path];
x_outer = [track2.outer(1,:), track2.outer(1,:)];
y_outer = [track2.outer(2,:), track2.outer(2,:)];
ref = [x_ref; y_ref; zeros(2, length(x_ref))];

% Calculate orientation matrices (for boundary constraint)
G_list = cell(1, length(x_ref));
for i = 1:length(x_ref)
    G_list{i} = generate_orient_matrix([x_outer(i); 
        y_outer(i)], [x_ref(i); y_ref(i)]);
end

% Edge of track
track_xmin = min([track2.inner(1,:), track2.outer(1,:)]);
track_xmax = max([track2.inner(1,:), track2.outer(1,:)]);
track_ymin = min([track2.inner(2,:), track2.outer(2,:)]);
track_ymax = max([track2.inner(2,:), track2.outer(2,:)]);

% Set initial state
xi = [x_ref(1),y_ref(1),init_v, ...
    atan((y_ref(2) - y_ref(1))/(x_ref(2) - x_ref(1)))]';

% Determine initial guess for inputs/states using over horizon using
% nonlinear MPC
solutions = controller_init{ref(:,1:1+N),[G_list{1:1+N}],xi};
U = solutions{1}; % Solved inputs along horizon
Xi = solutions{2}; % Solved states along horizon

history_U = []; % History of implemented inputs
history_Xi = xi; % History of implemented states

% Set simulation time
start_time = 0;
end_time = T*length(x_path);
times = start_time:T:end_time;

figure(1)
% Loop through times for simulation
tspan = [0 0];
fprintf('Simulation Progress:         ')
eps_sim = []; % Record array of epsilons, corresponding to norm of 
% difference between inputs in current and previous iteration. Indicates
% the convergence of inputs in an iteration.
for i = 1:length(times)-1 % starting off at first time already (not zeroth)
    fprintf('\b\b\b\b\b\b\b\b%6.2f %%',(i/(length(times)-1)*100))
    % Optimise for input via MPC
    U_omega_history_curr_step = U(1,1:N);
    U_a_history_curr_step = U(2,1:N);
    
    % Calculate predicted states
    tspan_pred = tspan;
    for n = 1:N
        Xi(:,n+1) = unicycle_approx_next_state(Xi(:,n),U(:,n),T);
    end
    
    for j = 1:M
        % Take average of inputs inside iteration to be next guess for U
        U(1,1:N) = mean(U_omega_history_curr_step, 1);
        U(2,1:N) = mean(U_a_history_curr_step, 1);

        % Linearise state-space matrices around current state
        [A,B] = gen_unicycle_state_input_cell(Xi,N,T);

        % Solve for U and Xi using MPC
        solutions = controller_sim{U,Xi,ref(:,i:i+N),[A{:}],[B{:}],[G_list{i:i+N}]};
        U = solutions{1};
        Xi = solutions{2};

        % Record epsilon over iterations
        eps_curr = norm([U(1,1:N) - U_omega_history_curr_step(j,1:N);
            U(2,1:N) - U_a_history_curr_step(j,1:N)], 2);

        % Record U over iterations
        U_omega_history_curr_step = [U_omega_history_curr_step; U(1,1:N)];
        U_a_history_curr_step = [U_a_history_curr_step; U(2,1:N)];
    end
    eps_sim = [eps_sim eps_curr];

    % Apply input to plant
    tspan = [tspan(2) tspan(2)+T];
    [t,x_ode] = ode45(@(t,x_ode) ode_unicycle_fun_a_input(t,x_ode,U(:,1)),...
        tspan, xi);
    xi = x_ode(length(x_ode),:)';
    
    % Update history
    history_U = [history_U,U(:,1)];
    history_Xi = [history_Xi,xi];
    
    % Plot position inside simulation
    if plot_vehicle_each_step
        plot_position_in_sim
    end
    
    % Prepare guess of inputs and states along horizon for next iteration
    U = [U(:,2:N), zeros(2,1)]; % keep last one as zero for now
    Xi = [xi, zeros(length(xi), N)];
end
% Average of final input differences at each time
eps_avg = mean(eps_sim);
fprintf('\n')

%% Plot System, States and Inputs
plot_positions
plot_states
plot_inputs

%% Plot Convergence 
% Check last epsilon value for each iteration over the simulation
figure
plot(times(1:length(eps_sim)), eps_sim)
eps_avg % Check average value of epsilon over simulation
