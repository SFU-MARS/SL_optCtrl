% Generate data for sampling initial state

nsamps = 200; % number of samples

% Starting circle
start_region = struct;
start_region.x = 3.2;
start_region.z = 3;
start_region.r = 0.5;

% Maximum and minimum states
vxmin = -2;
vxmax = 2;
vzmin = -2;
vzmax = 2;
phimin = -pi;
phimax = pi;
wmin = -pi;
wmax = pi;

init_states = struct;
init_states.x = zeros(1,nsamps);
init_states.vx = zeros(1,nsamps);
init_states.z = zeros(1,nsamps);
init_states.vz = zeros(1,nsamps);
init_states.phi = zeros(1,nsamps);
init_states.w = zeros(1,nsamps);

for i = 1:nsamps
    % sample starting position
    [x,z] = sample_circle(start_region.x,start_region.z,start_region.r);
    init_states.x(i) = x;
    init_states.z(i) = z;

    % sample starting velocities
    init_states.vx(i) = unifrnd(vxmin, vxmax); % vx
    init_states.vz(i) = unifrnd(vzmin, vzmax); % vy
    init_states.phi(i) = unifrnd(phimin, phimax); % phi
    init_states.w(i) = unifrnd(wmin, wmax); % w
    
end