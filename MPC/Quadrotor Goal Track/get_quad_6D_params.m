function obj = get_quad_6D_params
    obj = struct;

    T1Max = 36.7875/2;  
    T1Min = 0;
    T2Max = 36.7875/2;  
    T2Min =0;
    m = 1.25; %kg
    grav = 9.81;
    transDrag = 0.25;
    rotDrag = 0.02255;
    l = 0.5; %meters
    Iyy = 0.03;

    dims = [1 2 5 6];

    % Basic vehicle properties
    obj.pdim = [1 3]; % Position dimensions
    obj.hdim = 5;   % Heading dimensions
    obj.nx = length(dims);
    obj.nu = 2;  

%     obj.x = x; Ignore this... all we need are model params
%     obj.xhist = obj.x;
    obj.dims = dims;

    obj.T1Max = T1Max;
    obj.T1Min = T1Min;
    obj.T2Max = T2Max;
    obj.T2Min = T2Min;
    obj.m = m;
    obj.grav = grav;
    obj.transDrag = transDrag;
    obj.rotDrag = rotDrag;
    obj.Iyy = Iyy;
    obj.l = l;
end
