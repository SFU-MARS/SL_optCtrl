function [ dxdt ] = ode_unicycle_fun_a_input( t, x, u )
%ODEFUN ODE for unicycle
    dxdt = zeros(3,1);
    dxdt(1) = x(3)*cos(x(4));
    dxdt(2) = x(3)*sin(x(4));
    dxdt(3) = u(2);
    dxdt(4) = u(1);
end

