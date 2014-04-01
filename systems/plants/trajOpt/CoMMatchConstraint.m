classdef CoMMatchConstraint < NonlinearConstraint
  % with input argument kinsol and com, enforce the constraint that com =
  % robot_com(kinsol), where robot_com is the function to compute the CoM position from
  % the kinematics tree.
  properties(SetAccess = protected)
    robot
  end
  
  methods
    function obj = CoMMatchConstraint(robot)
      obj = obj@NonlinearConstraint(zeros(3,1),zeros(3,1),robot.getNumDOF+3);
      obj.robot = robot;
      nq = obj.robot.getNumDOF;
      obj = obj.setSparseStructure([reshape(bsxfun(@times,(1:3)',ones(1,nq)),[],1);1;2;3],...
        [reshape(bsxfun(@times,ones(3,1),(1:nq)),[],1);nq+1;nq+2;nq+3]);
    end
    
    function [c,dc] = eval(obj,kinsol,com)
      [com_kinsol,J] = obj.robot.getCOM(kinsol);
      c = com_kinsol-com;
      dc = [J -eye(3)];
    end
  end
end