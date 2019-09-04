function draw_goal(goal)
%DRAW_GOAL
%   Draw goal region in xy plane
x_left = goal.xmin;
x_right = goal.xmax;
z_bottom = goal.zmin;
z_top = goal.zmax;
grey = [0.87 0.87 0.87]; % color vector
fill([x_left,x_right,x_right,x_left],[z_bottom,z_bottom,z_top,z_top],grey);
end

