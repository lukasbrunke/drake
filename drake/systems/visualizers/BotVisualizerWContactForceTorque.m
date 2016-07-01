classdef BotVisualizerWContactForceTorque < BotVisualizer
  properties(SetAccess = protected)
    kinframes
    num_forcetorque_sensor
  end
  
  methods
    function obj = BotVisualizerWContactForceTorque(manip,use_collision_geometry,kinframes)
      obj = obj@BotVisualizer(manip,use_collision_geometry);
      obj.kinframes = kinframes;
      obj.num_forcetorque_sensor = length(obj.kinframes);
    end
    
    function draw(obj,t,y)
      if(length(y) ~= obj.model.num_positions+obj.model.num_velocities+6*obj.num_forcetorque_sensor)
        error('y is not of correct size');
      end
      draw@BotVisualizer(obj,t,y);
      force_torques = reshape(y(obj.model.num_positions+obj.model.num_velocities+1:end),6,obj.num_forcetorque_sensor);
      q = y(1:obj.model.num_positions);
      kinsol = obj.model.doKinematics(q);
      lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),sprintf(' force'));
      for i = 1:obj.num_forcetorque_sensor
        sensor_pos_quat = obj.model.forwardKin(kinsol,obj.kinframes(i),zeros(3,1),2);
        force_world = quat2rotmat(sensor_pos_quat(4:7))*force_torques(1:3,i);
        
        
        
        lcmgl.glColor3f(1,0,0);
        lcmgl.drawVector(sensor_pos_quat(1:3),force_world/5000,0.01,0.02,0.02);
        
        
      end
      lcmgl.switchBuffers();
    end
  end
end