classdef SearchContactsFixedDisturbanceFullKinematicsSOSPlanner < SearchContactsComDynamicsFullKinematicsSOSPlanner
  properties(SetAccess = protected)
    disturbance_pos % A 3 x obj.N matrix
  end
  
  methods
    function obj = SearchContactsFixedDisturbanceFullKinematicsSOSPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,options)
      if(nargin<12)
        options = struct();
      end
      obj = obj@SearchContactsComDynamicsFullKinematicsSOSPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,options);
      sizecheck(disturbance_pos,[3,obj.N]);
      obj.disturbance_pos = disturbance_pos;
      
      obj = obj.addSOScondition();
    end
  end
end