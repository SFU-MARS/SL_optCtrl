function [value] = DubinsCar_approx_RS(gMin, gMax, gN, wMax, speed, tMax, interval, target)

    global gXYT;

    pdDims = 3;               % 3rd dimension is periodic
    gXYT = createGrid(gMin, gMax, gN, pdDims);

    tau = 0:interval:tMax;

    uMode = 'min';

    % Define dynamic system
    % obj = DubinsCar(x, wMax, speed, dMax)
    dCar = DubinsCar([0,0,0], wMax, speed);

    % Put grid and dynamic systems into schemeData
    schemeData.grid = gXYT;
    schemeData.dynSys = dCar;
    schemeData.accuracy = 'high';
    schemeData.uMode = uMode;

    %% Compute value function
    HJIextraArgs.visualize = false; %show plot
    HJIextraArgs.fig_num = 1; %set figure number
    HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update

    %[data, tau, extraOuts] = ...
    % HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
    [value, tau2, ~] = HJIPDE_solve(target, tau, schemeData, 'none', HJIextraArgs);

