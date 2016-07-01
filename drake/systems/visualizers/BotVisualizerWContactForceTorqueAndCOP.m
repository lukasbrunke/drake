classdef BotVisualizerWContactForceTorqueAndCOP < BotVisualizer
  properties(SetAccess = protected)
    kinframes
    num_forcetorque_sensor
    contact_body
    contact_pts
    num_contact_body
  end
  
  methods
    function obj = BotVisualizerWContactForceTorqueAndCOP(manip,use_collision_geometry,kinframes,contact_body)
      obj = obj@BotVisualizer(manip,use_collision_geometry);
      obj.kinframes = kinframes;
      obj.num_forcetorque_sensor = length(obj.kinframes);
      obj.contact_body = contact_body;
      obj.num_contact_body = length(obj.contact_body);
      obj.contact_pts = cell(obj.num_contact_body,1);
      for i = 1:obj.num_contact_body
        obj.contact_pts{i} = manip.getBody(obj.contact_body(i)).getTerrainContactPoints();
      end
    end
    
    function draw(obj,t,y)
      if(length(y) ~= obj.model.num_positions+obj.model.num_velocities+6*obj.num_forcetorque_sensor)
        error('y is not of correct size');
      end
      draw@BotVisualizer(obj,t,y);
      force_torques = reshape(y(obj.model.num_positions+obj.model.num_velocities+1:end),6,obj.num_forcetorque_sensor);
      q = y(1:obj.model.num_positions);
      kinsol = obj.model.doKinematics(q);
      contact_pos = cell(1,obj.num_contact_body);
      for i = 1:obj.num_contact_body
        contact_pos{i} = obj.model.forwardKin(kinsol,obj.contact_body(i),obj.contact_pts{i},0);
      end
      all_contact_pos = cell2mat(contact_pos);
      
      
      lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),'cop');
      if(~isempty(all_contact_pos))
        ground_contact_pos = all_contact_pos(:,all_contact_pos(3,:)<2e-3);
        if(size(ground_contact_pos,2)>=3)
          K = convhull(ground_contact_pos(1,:),ground_contact_pos(2,:));
          lcmgl.glColor3f(0,0,1);
          lcmgl.polygon(ground_contact_pos(1,K(1:end-1)),ground_contact_pos(2,K(1:end-1)),zeros(1,length(K)-1));
        end
      end
      force_world = zeros(3,obj.num_forcetorque_sensor);
      torque_world = zeros(3,obj.num_forcetorque_sensor);
      sensor_pos = zeros(3,obj.num_forcetorque_sensor);
      for i = 1:obj.num_forcetorque_sensor
        sensor_pos_quat = obj.model.forwardKin(kinsol,obj.kinframes(i),zeros(3,1),2);
        sensor_rotmat = quat2rotmat(sensor_pos_quat(4:7));
        force_world(:,i) = sensor_rotmat*force_torques(1:3,i);
        torque_world(:,i) = sensor_rotmat*force_torques(4:6,i);
        sensor_pos(:,i) = sensor_pos_quat(1:3);
      end
      total_torque = sum(cross(sensor_pos,force_world,1),2)+sum(torque_world,2);
      total_force = sum(force_world,2);
      cop_pos = [-total_torque(2);total_torque(1)]/total_force(3);
      lcmgl.glColor3f(1,0,0);
      lcmgl.sphere([cop_pos;0],0.02,20,20);
      lcmgl.switchBuffers();

    end
  end
end