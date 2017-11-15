function TestSDPBilinearIKrelaxation()
p = RigidBodyManipulator([getDrakePath(), '/examples/IRB140/urdf/irb_140.urdf'], struct('floating', false));

ee_idx = p.findLinkId('link_6');
ik = InverseKinematicsBMI(p);
ik.itr_max = 1;
ik.trial_max = 1;

sample_per_axis = 21;
x_pos_sample = linspace(0, 1, sample_per_axis);
y_pos_sample = linspace(-0.5, 0.5, sample_per_axis);
z_pos_sample = linspace(-0.1, 0.9, sample_per_axis);

q_random = randn(6, 20);
bmi_ik_infeasible_count = 0;
nlp_success_with_bmi_sol = 0;
nlp_success = 0;
for x = x_pos_sample
  for y = y_pos_sample
    for z = z_pos_sample
      ik_ijk = ik.fixLinkPosture(ee_idx, [x;y;z], [1;0;0;0]);
      [solver_sol, info] = ik_ijk.optimize();
      if (info == 0)
        bmi_ik_infeasible_count = bmi_ik_infeasible_count + 1;
      else
        sol = ik_ijk.retrieveSolution(solver_sol);
        position_cnstr = WorldPositionConstraint(p, ee_idx, [0;0;0], [x;y;z], [x;y;z]);
        orient_cnstr = WorldQuatConstraint(p, ee_idx, [1;0;0;0], 0);
        [q, info_nlp_with_bmi_sol] = inverseKin(p, sol.q, zeros(6, 1), position_cnstr, orient_cnstr);
        if (info_nlp_with_bmi_sol < 10)
          nlp_success_with_bmi_sol = nlp_success_with_bmi_sol + 1;
          nlp_success = nlp_success + 1;
        else
          for j = 1 : size(q_random, 2)
            [~, info_nlp] = inverseKin(p, q_random(:, j), zeros(6, 1), position_cnstr);
            if (info_nlp < 10)
              nlp_success = nlp_success + 1;
              break
            end
          end
        end
      end
    end
  end
end

keyboard
end