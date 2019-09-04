function [A_o,boundary,b_o] = generate_rectangle_obstacle_constraint_mats(rect_obs)
%GENERATE_OBSTACLE_CONSTRAINT_MATS
%   Generates A_o and b_o matrix for rectangular obstacles
RECT_SIDES = 4;
A_o = [-1 0; 1 0; 0 -1; 0 1];
boundary = [rect_obs.center_x-rect_obs.width/2,...
        rect_obs.center_x+rect_obs.width/2,...
        rect_obs.center_z-rect_obs.height/2,...
        rect_obs.center_z+rect_obs.height/2];
b_o = diag([-1 1 -1 1])*boundary';