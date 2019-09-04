function [x_samp,y_samp] = sample_circle(x_c,y_c,r)
%SAMPLE_CIRCLE
%   Uniformly sample from circle
r_samp = unifrnd(0,r);
theta_samp = unifrnd(0,2*pi);
x_samp = r_samp*cos(theta_samp) + x_c;
y_samp = r_samp*sin(theta_samp) + y_c;
end

