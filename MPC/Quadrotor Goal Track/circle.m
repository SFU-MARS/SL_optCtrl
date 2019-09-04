function circle(x,y,r,color)
% CIRCLE
%   Plots circle of specific radius at specific point
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
plot(xunit, yunit, color)
