classdef SimpleCentroidalTorqueConstraint < NonlinearConstraint
  % require the centroidal torque computed from external force equal to zero
  properties(SetAccess = protected)
    num_F
    contact_wrench_constraint
    t
  end
  
  properties(Access = protected)
    num_cwcnstr % number of ContactWrenchConstraint
    nq
  end
  
  methods
    function obj = SimpleCentroidalTorqueConstraint(t,nq,num_F,varargin)
      % obj =
      % CentroidalWrenchConstraint(num_F,contact_wrench_constraint1,contact_wrench_constraint2,...)
      % @param contact_wrench_constraint. A ContactWrenchConstraint object. This provide
      % the information about the active contact 
      % @param t       -- A scalar. The time to enforce this constraint
      % @param nq      -- A scalar. nq = robot.getNumDOF;
      % @param num_F   -- The total number of contact force parameters in all
      % ContactWrenchConstraint
      obj = obj@NonlinearConstraint(zeros(3,1),zeros(3,1),nq+num_F);
      obj.t = t;
      obj.nq = nq;
      obj.num_F = num_F;
      obj.contact_wrench_constraint = varargin;
      obj.num_cwcnstr = nargin-3;
      count_F = 0;
      for i = 1:obj.num_cwcnstr
        if(~isa(varargin{i},'ContactWrenchConstraint'))
          error('Drake:CentroidalWrenchConstraint:The input should be a ContactWrenchConstraint object');
        end
        if(~varargin{i}.isTimeValid(t))
          error('Drake:CentroidalWrenchConstraint:Please only pass the ContactWrenchConstraint that is active at time t');
        end
        count_F = count_F+prod(varargin{i}.F_size);
      end
      if(count_F ~= num_F)
        error('Drake:CentroidalWrenchConstraint: num_F does not match with ContactWrenchConstraint');
      end
    end
    
    function [c,dc] = eval(obj,kinsol,x_nonkinsol)
      c = zeros(3,1);
      dc = zeros(3,obj.nq+obj.num_F);
      count_F = 0;
      for i = 1:obj.num_cwcnstr
        num_F_i = prod(obj.contact_wrench_constraint{i}.F_size);
        F_i = x_nonkinsol(count_F+(1:num_F_i));
        [torque_i,dtorque_i] = obj.contact_wrench_constraint{i}.centroidalTorque(obj.t,kinsol,F_i);
        c = c+torque_i;
        dc(:,1:obj.nq) = dc(:,1:obj.nq)+dtorque_i(:,1:obj.nq);
        dc(:,obj.nq+count_F+(1:num_F_i)) = dtorque_i(:,obj.nq+1:end);
        count_F = count_F+num_F_i;
      end
    end
  end
end