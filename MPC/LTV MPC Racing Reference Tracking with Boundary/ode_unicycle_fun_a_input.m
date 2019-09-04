function [ dxidt ] = ode_unicycle_fun_a_input( t, xi, u )
%ODEFUN ODE
% ODE function for unicycle model, modified so angular velocity and
% acceleration are the inputs
% Inputs
% t: times (required for ode solver)
% xi: states of system [x, y, v, theta]
% u: inputs in system [omega, a]
% Outputs
% dxidt: time derivative of states
dxidt = zeros(3,1);
dxidt(1) = xi(3)*cos(xi(4));
dxidt(2) = xi(3)*sin(xi(4));
dxidt(3) = u(2);
dxidt(4) = u(1);
end

