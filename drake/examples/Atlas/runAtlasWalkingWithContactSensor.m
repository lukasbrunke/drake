function runAtlasWalkingWithContactSensor(robot_options, walking_options)
% Run the new split QP controller, which consists of separate PlanEval
% and InstantaneousQPController objects. The controller will also
% automatically transition to standing when it reaches the end of its walking
% plan.

checkDependency('gurobi');
checkDependency('lcmgl');

if nargin < 1; robot_options = struct(); end;
if nargin < 2; walking_options = struct(); end;

robot_options = applyDefaults(robot_options, struct('use_bullet', true,...
                                                    'terrain', RigidBodyFlatTerrain,...
                                                    'floating', true,...
                                                    'ignore_self_collisions', true,...
                                                    'ignore_friction', true,...
                                                    'enable_fastqp', false,...
                                                    'urdf_modifications_file', fullfile(getDrakePath(), 'examples', 'Atlas', 'config', 'urdf_modifications_no_hands.yaml'),...
                                                    'dt', 0.001));
% silence some warnings
warning('off','Drake:RigidBodyManipulator:UnsupportedContactPoints')
warning('off','Drake:RigidBodyManipulator:UnsupportedVelocityLimits')

% construct robot model
r = Atlas(fullfile(getDrakePath,'examples','Atlas','urdf','atlas_minimal_contact.urdf'),robot_options);
r = r.removeCollisionGroupsExcept({'heel','toe'});
r = compile(r);

r_pure = r;
l_foot_body = findLinkId(r,'l_foot');
% l_foot_frame1 = RigidBodyFrame(l_foot_body,l_toe(:,1),zeros(3,1),'l_foot_force
l_foot_frame = RigidBodyFrame(l_foot_body,zeros(3,1),zeros(3,1),'l_foot_forcetorque');
r = r.addFrame(l_foot_frame);
l_foot_force_sensor = ContactForceTorqueSensor(r, l_foot_frame);
r = addSensor(r, l_foot_force_sensor);
r_foot_body = findLinkId(r,'r_foot');
r_foot_frame = RigidBodyFrame(r_foot_body,zeros(3,1),zeros(3,1),'r_foot_forcetorque');
r = r.addFrame(r_foot_frame);
r_foot_force_sensor = ContactForceTorqueSensor(r, r_foot_frame);
r = addSensor(r, r_foot_force_sensor);

r = compile(r);

walking_options = applyDefaults(walking_options, struct('initial_pose', [],...
                                                        'navgoal', [0.5;0;0;0;0;0],...
                                                        'max_num_steps', 6,...
                                                        'rms_com_tolerance', 0.0051,...
                                                        'urdf_modifications_file', ''));
walking_options = applyDefaults(walking_options, r.default_footstep_params);
walking_options = applyDefaults(walking_options, r.default_walking_params);

% set initial state to fixed point
load(r.fixed_point_file, 'xstar');
if ~isempty(walking_options.initial_pose), xstar(1:6) = walking_options.initial_pose; end
xstar = r.resolveConstraints(xstar);
r = r.setInitialState(xstar);

v = r.constructVisualizer;
v.display_dt = 0.01;

nq = getNumPositions(r);

x0 = xstar;

% Find the initial positions of the feet
R=rotz(walking_options.navgoal(6));

rfoot_navgoal = walking_options.navgoal;
lfoot_navgoal = walking_options.navgoal;

rfoot_navgoal(1:3) = rfoot_navgoal(1:3) + R*[0;-0.13;0];
lfoot_navgoal(1:3) = lfoot_navgoal(1:3) + R*[0;0.13;0];

% Plan footsteps to the goal
goal_pos = struct('right', rfoot_navgoal, 'left', lfoot_navgoal);
footstep_plan = r.planFootsteps(x0(1:nq), goal_pos, [], struct('step_params', walking_options));
for j = 1:length(footstep_plan.footsteps)
  footstep_plan.footsteps(j).walking_params = walking_options;
end

% Generate a dynamic walking plan
walking_plan_data = r.planWalkingZMP(x0(1:r.getNumPositions()), footstep_plan);

x0 = r.getInitialState();

% Build our controller and plan eval objects
control = bipedControllers.InstantaneousQPController(r_pure.getManipulator().urdf{1}, r.control_config_file, walking_options.urdf_modifications_file);
planeval = bipedControllers.BipedPlanEval(r_pure, walking_plan_data);
plancontroller = bipedControllers.BipedPlanEvalAndControlSystem(r_pure, control, planeval);
sys = feedback(r, plancontroller);

% Pass through outputs from robot
outs(1).system = 1;
outs(1).output = 1;
outs(2).system = 1;
outs(2).output = 2;
outs(3).system = 1;
outs(3).output = 3;
sys = mimoFeedback(r, plancontroller, [], [], [], outs);

% Add a visualizer
v = r.constructVisualizer;
v.display_dt = 0.01;
S=warning('off','Drake:DrakeSystem:UnsupportedSampleTime');
output_select(1).system=1;
output_select(1).output=1;
output_select(2).system=1;
output_select(2).output=2;
output_select(3).system=1;
output_select(3).output=3;
sys = mimoCascade(sys,v,[],[],output_select);
warning(S);

% Simulate and draw the result
T = walking_plan_data.duration + 1;
ytraj = simulate(sys, [0, T], x0, walking_options);
[com, rms_com] = r.plotWalkingTraj(ytraj, walking_plan_data);

l_foot_forcetorque = r.findFrameId('l_foot_forcetorque');
r_foot_forcetorque = r.findFrameId('r_foot_forcetorque');
v2 = BotVisualizerWContactForceTorque(r.getManipulator,false,[l_foot_forcetorque r_foot_forcetorque]);
v2.playback(ytraj,struct('slider',true));

if ~rangecheck(rms_com, 0, walking_options.rms_com_tolerance);
  error('Drake:runWalkingDemo:BadCoMTracking', 'Center-of-mass during execution differs substantially from the plan.');
end