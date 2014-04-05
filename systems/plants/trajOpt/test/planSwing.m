function [swing_ts, swing_poses, takeoff_time, landing_time] = planSwing(biped, last_pos, next_pos, options)
% Compute a collision-free swing trajectory for a single foot. Uses the biped's RigidBodyTerrain to compute a slice of terrain between the two poses.

if ~isfield(options, 'step_speed')
  options.step_speed = 0.5; % m/s
end
if ~isfield(options, 'step_height')
  options.step_height = biped.nom_step_clearance; %m
end
if ~isfield(options, 'ignore_terrain')
  options.ignore_terrain = false;
end

if options.step_speed < 0
  % negative step speed is an indicator to take a fast, fixed-duration step (e.g. for recovery)
  fixed_duration = -options.step_speed;
  options.step_speed = 1;
else
  fixed_duration = 0;
end

debug = false;

ignore_height = 0.5; % m, height above which we'll assume that our heightmap is giving us bad data (e.g. returns from an object the robot is carrying)
hold_frac = 0.2; % fraction of leg swing time spent shifting weight to stance leg
min_hold_time = 0.1; % s
pre_contact_height = 0.005; % height above the ground to aim for when foot is landing
foot_yaw_rate = 0.75; % rad/s

step_dist_xy = sqrt(sum((next_pos(1:2) - last_pos(1:2)).^2));

if fixed_duration
  apex_fracs = [0.15,0.16, 0.84,0.85];
else
  apex_fracs = [0.15, 0.85];
end

next_pos(4:6) = last_pos(4:6) + angleDiff(last_pos(4:6), next_pos(4:6));
apex_pos = interp1([0, 1], [last_pos, next_pos]', apex_fracs)';
apex_pos(3,:) = last_pos(3) + options.step_height + max([next_pos(3) - last_pos(3), 0]);

apex_pos_l = [apex_fracs * step_dist_xy; apex_pos(3,:)];


  %% Just ignore the terrain and provide an apex pose
  step_dist_xy = 1;
  traj_pts = [[0; last_pos(3)], [apex_fracs; apex_pos(3,:)], [1;next_pos(3) + pre_contact_height]];

traj_pts_xyz = [last_pos(1) + (next_pos(1) - last_pos(1)) * traj_pts(1,:) / step_dist_xy;
                last_pos(2) + (next_pos(2) - last_pos(2)) * traj_pts(1,:) / step_dist_xy;
                traj_pts(2,:)];


%% Compute time required for swing from cartesian distance of poses as well as yaw distance
d_dist = sqrt(sum(diff(traj_pts_xyz, 1, 2).^2, 1));
total_dist_traveled = sum(d_dist);
traj_dts = max([d_dist / options.step_speed;
                (d_dist / total_dist_traveled) .* (abs(angleDiff(next_pos(6), last_pos(6))) / foot_yaw_rate)],[],1);
traj_ts = [0, cumsum(traj_dts)] ;

%% Add time to shift weight
if fixed_duration
  hold_time = 0.1;
  traj_ts = traj_ts * ((fixed_duration - 2*hold_time) / traj_ts(end));
  traj_ts = [0, traj_ts+hold_time, traj_ts(end) + 2*hold_time];
else
  hold_time = traj_ts(end) * hold_frac;
  hold_time = max([hold_time, min_hold_time]);
  traj_ts = [0, traj_ts + hold_time, traj_ts(end) + 2.5*hold_time]; % add time for weight shift
end

traj_pts_xyz = [last_pos(1:3), traj_pts_xyz, next_pos(1:3)];
landing_time = traj_ts(end-1);
takeoff_time = traj_ts(2);

%% Interpolate in rpy to constrain the foot orientation. We may set these values to NaN later to free up the foot orientation
rpy_pts = [last_pos(4:6), interp1(traj_ts([2,end-1]), [last_pos(4:6), next_pos(4:6)]', traj_ts(2:end-1))', next_pos(4:6)];

swing_poses.center = [traj_pts_xyz; rpy_pts];

swing_ts = traj_ts;


if debug
  figure(1)
  clf
  hold on
  plot(terrain_pts(1,:), terrain_pts(2,:), 'g.')
  plot(expanded_terrain_pts(1,:), expanded_terrain_pts(2,:), 'ro')

  t = linspace(traj_ts(1), traj_ts(end));
  xyz = step_traj.eval(t);
  plot(sqrt(sum(bsxfun(@minus, xyz(1:2,:), xyz(1:2,1)).^2)), xyz(3,:),'k')
  axis equal
end

end

% function terrain_pts = terrainSample(biped, last_pos, next_pos, contact_width, nlambda, nrho)
%   step_dist_xy = sqrt(sum((next_pos(1:2) - last_pos(1:2)).^2));
%   lambda_hat = (next_pos(1:2) - last_pos(1:2)) / step_dist_xy;

%   rho_hat = [0, -1; 1, 0] * lambda_hat;

%   terrain_pts = zeros(2, nlambda);
%   lambdas = linspace(0, step_dist_xy, nlambda);
%   rhos = linspace(-contact_width, contact_width, nrho);
%   [R, L] = meshgrid(rhos, lambdas);
%   xy = bsxfun(@plus, last_pos(1:2), bsxfun(@times, reshape(R, 1, []), rho_hat) + bsxfun(@times, reshape(L, 1, []), lambda_hat));
%   z = medfilt2(reshape(biped.getTerrainHeight(xy), size(R)), 'symmetric');
% %   plot_lcm_points([xy; reshape(z, 1, [])]', repmat([1 0 1], size(xy, 2), 1), 101, 'Swing terrain pts', 1, 1);
%   terrain_pts(2, :) = max(z, [], 2);
%   terrain_pts(1,:) = lambdas;
% end
