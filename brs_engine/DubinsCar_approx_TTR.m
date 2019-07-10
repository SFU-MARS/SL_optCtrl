function [ttr_value] = ...
  DubinsCar_approx_TTR(gMin, gMax, gN, value, tMax, interval)

    global gXYT;

    pdDims = 3;               % 3rd dimension is periodic
    gXYT = createGrid(gMin, gMax, gN, pdDims);

    % Time horizon and intermediate results
    tau = 0:interval:tMax;

    ttr_value = TD2TTR(gXYT, value, tau);



end
