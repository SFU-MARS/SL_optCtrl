function x_next = ltv_predict_next_state(x,u,T)
%LTV PREDICT NEXT STATE
%  Solve for next state using Euler approximation
% dotx = x(3)*cos(x(4));
% doty = x(3)*sin(x(4));
% dotv = u(2);
% dottheta = u(1);
% x_next = zeros(4,1);
% x_next(1) = x(1) + T*dotx;
% x_next(2) = x(2) + T*doty;
% x_next(3) = x(3) + T*dotv;
% x_next(4) = x(4) + T*dottheta;

dotx = x(3)*cos(x(4));
ddotx = -x(3)*sin(x(4))*u(1) + cos(x(4))*u(2);
doty = x(3)*sin(x(4));
ddoty = x(3)*cos(x(4))*u(1) + sin(x(4))*u(2);
dotv = u(2);
dottheta = u(1);
x_next = zeros(4,1);
x_next(1) = x(1) + T*dotx + T^2/2*ddotx;
x_next(2) = x(2) + T*doty + T^2/2*ddoty;
x_next(3) = x(3) + T*dotv;
x_next(4) = x(4) + T*dottheta;
end

