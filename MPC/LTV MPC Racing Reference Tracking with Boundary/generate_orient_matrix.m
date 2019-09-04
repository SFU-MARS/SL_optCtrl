function [G] = generate_orient_matrix(p_out, p_centre)
%GENERATE_ORIENT_MATRIX
% Generates a matrix G that maps (x - x_cent,y - y_cent) to a coordinate
% space (x',y') where the constraint -1 <= x' <= 1 indicates that the the
% point (x,y) remains inside two parallel lines representing the boundary
% of the track.
% Inputs
% p_out: 2:1 array with x and y coordinate of point on track outer boundary
% p_center: 2x1 array with x and y coordinate of point on track center

% Obtain coordinate of p_out treating p_centre as the origin
p_out_centred = p_out - p_centre;

% Calculate G
G = 1/(p_out_centred(1)^2 + p_out_centred(2)^2)...
    * [p_out_centred(1), p_out_centred(2); 
    -p_out_centred(2), p_out_centred(1)];
end

