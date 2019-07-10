function [target] = ...
  DubinsCar_create_init_target(gMin, gMax, gN, goalState, goalRadii)
% Xubo Lyu, 2019-01-31

global gXYT;

%% Target and obstacles
% 3D grid limits (x, y, \theta)
if nargin < 3
  gMin = [-5; -5; -pi];
  gMax = [5; 5; pi];
  gN = 101*ones(3,1);
end
Xdim = 1;
Ydim = 2;
Tdim = 3;
pdDim = Tdim;
gXYT = createGrid(gMin, gMax, gN, pdDim);

R = goalRadii(Xdim)

%% Initial target set
target = shapeCylinder(gXYT, 3, goalState, R);

end
