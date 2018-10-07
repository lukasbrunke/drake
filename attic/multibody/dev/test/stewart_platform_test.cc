#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "drake/math/evenly_distributed_pts_on_sphere.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mixed_integer_optimization_util.h"
#include "drake/solvers/mixed_integer_rotation_constraint.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/rotation_constraint.h"

namespace drake {
namespace multibody {
void RemoveFileIfExist(const std::string& file_name) {
  std::ifstream file(file_name);
  if (file) {
    if (remove(file_name.c_str()) != 0) {
      throw std::runtime_error("Error deleting file " + file_name);
    }
  }
  file.close();
}

template <typename Derived>
void WriteRowVectorToFile(const Derived& value,
                          const std::string& out_file_name) {
  std::fstream output_file;
  output_file.open(out_file_name, std::ios::app | std::ios::out);
  if (output_file.is_open()) {
    for (int i = 0; i < value.size(); ++i) {
      output_file << value(i) << " ";
    }
    output_file << "\n";
  }
  output_file.close();
}

void AddLeg(const Eigen::Vector3d& p_PBi, const Eigen::Vector3d& p_BAi,
            double leg_length, const Matrix3<symbolic::Variable>& R_BP,
            const Vector3<symbolic::Variable>& p_BPo,
            const std::array<Eigen::VectorXd, 3>& phi_p_BPo,
            const std::array<VectorX<symbolic::Variable>, 3>& lambda_p_BPo,
            const Matrix3<symbolic::Expression>& p_BPo_times_R_BP,
            bool use_linear_constraint, solvers::MathematicalProgram* prog) {
  // First add the constraint |p_BPo + R_BP * p_PBi - p_BAi| <= leg_length
  const Vector3<symbolic::Expression> p_AiBi_B = p_BPo + R_BP * p_PBi - p_BAi;
  if (use_linear_constraint) {
    std::vector<Eigen::Vector3d> unit_vecs;
    unit_vecs.emplace_back(1, 0, 0);
    unit_vecs.emplace_back(0, 1, 0);
    unit_vecs.emplace_back(0, 0, 1);
    unit_vecs.emplace_back(std::sqrt(2) / 2, std::sqrt(2) / 2, 0);
    unit_vecs.emplace_back(std::sqrt(2) / 2, -std::sqrt(2) / 2, 0);
    unit_vecs.emplace_back(std::sqrt(2) / 2, 0, std::sqrt(2) / 2);
    unit_vecs.emplace_back(std::sqrt(2) / 2, 0, -std::sqrt(2) / 2);
    unit_vecs.emplace_back(0, std::sqrt(2) / 2, std::sqrt(2) / 2);
    unit_vecs.emplace_back(0, std::sqrt(2) / 2, -std::sqrt(2) / 2);
    unit_vecs.emplace_back(1 / std::sqrt(3), 1 / std::sqrt(3),
                           1 / std::sqrt(3));
    unit_vecs.emplace_back(1 / std::sqrt(3), -1 / std::sqrt(3),
                           1 / std::sqrt(3));
    unit_vecs.emplace_back(1 / std::sqrt(3), 1 / std::sqrt(3),
                           -1 / std::sqrt(3));
    unit_vecs.emplace_back(1 / std::sqrt(3), 1 / std::sqrt(3),
                           -1 / std::sqrt(3));
    for (const auto& unit_vec : unit_vecs) {
      prog->AddLinearConstraint(unit_vec.dot(p_AiBi_B) <= leg_length);
      prog->AddLinearConstraint(-unit_vec.dot(p_AiBi_B) <= leg_length);
    }

    const Eigen::Matrix3Xd unit_vectors =
        math::UniformPtsOnSphereFibonacci(200);
    for (int i = 0; i < unit_vectors.cols(); ++i) {
      prog->AddLinearConstraint(unit_vectors.col(i).dot(p_AiBi_B) <=
                                leg_length);
    }
  } else {
    Vector4<symbolic::Expression> v_lorentz;
    v_lorentz(0) = leg_length;
    v_lorentz.tail<3>() = p_AiBi_B;
    prog->AddLorentzConeConstraint(v_lorentz);
  }
  // The constraint |p_BPo + R_BP * p_PBi - p_BAi| >= leg_length is equivalent
  // to
  // p_PBiᵀ * p_PBi + 2 (p_BPo - p_BAi)ᵀ * R_BP * p_PBi + (p_BPo - p_BAi)ᵀ *
  // (p_BPo - p_BAi) >= leg_length²
  //
  symbolic::Expression lhs =
      p_PBi.squaredNorm() -
      2 * p_BAi.dot(R_BP.cast<symbolic::Expression>() * p_PBi) +
      p_BAi.squaredNorm() - 2 * p_BPo.cast<symbolic::Expression>().dot(p_BAi);
  // Now we need to add an upper bound of p_BPoᵀ * p_BPo
  // If x = φᵀλ, then x² <= sum_i φ(i)² * λ(i)
  symbolic::Expression p_BPo_square_upperbound{0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < phi_p_BPo[i].size(); ++j) {
      p_BPo_square_upperbound +=
          phi_p_BPo[i](j) * phi_p_BPo[i](j) * lambda_p_BPo[i](j);
    }
  }
  lhs += p_BPo_square_upperbound;

  // Now add 2 * p_BPoᵀ * R_BP * p_PBi
  const RowVector3<symbolic::Expression> p_BPo_transpose_times_R_BP =
      p_BPo_times_R_BP.cast<symbolic::Expression>().colwise().sum();
  lhs += (2 * p_BPo_transpose_times_R_BP * p_PBi)(0);

  prog->AddLinearConstraint(lhs >= leg_length * leg_length);
}

int DoMain() {
  solvers::MathematicalProgram prog;
  auto R_BP = solvers::NewRotationMatrixVars(&prog, "R");
  bool use_linear_constraint = true;
  if (!use_linear_constraint) {
    solvers::AddRotationMatrixOrthonormalSocpConstraint(&prog, R_BP);
  }
  solvers::MixedIntegerRotationConstraintGenerator rotation_generator(
      solvers::MixedIntegerRotationConstraintGenerator::Approach::
          kBilinearMcCormick,
      8, solvers::IntervalBinning::kLogarithmic);
  const auto R_BP_return = rotation_generator.AddToProgram(R_BP, &prog);
  auto p_BPo = prog.NewContinuousVariables<3>("p");

  // We will have bilinear product p_BPo(i) * R_BP.col(j)(i) and the quadratic
  // product p_BPo(i) * p_BPo(i)
  std::array<Eigen::VectorXd, 3> phi_p_BPo;
  phi_p_BPo[0] = Eigen::VectorXd::LinSpaced(17, -0.5, 1.5);
  phi_p_BPo[1] = Eigen::VectorXd::LinSpaced(17, -1, 1);
  phi_p_BPo[2] = Eigen::VectorXd::LinSpaced(17, -0.5, 1);

  std::array<VectorX<symbolic::Variable>, 3> lambda_p_BPo;
  std::array<VectorX<symbolic::Variable>, 3> b_p_BPo;
  for (int i = 0; i < 3; ++i) {
    lambda_p_BPo[i] = prog.NewContinuousVariables(phi_p_BPo[i].size());
    b_p_BPo[i] = solvers::AddLogarithmicSos2Constraint(
        &prog, lambda_p_BPo[i].cast<symbolic::Expression>());
    prog.AddLinearConstraint(lambda_p_BPo[i].dot(phi_p_BPo[i]) == p_BPo(i));
  }

  auto p_BPo_times_R_BP = prog.NewContinuousVariables<3, 3>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      solvers::AddBilinearProductMcCormickEnvelopeSos2(
          &prog, p_BPo(i), R_BP(i, j),
          symbolic::Expression(p_BPo_times_R_BP(i, j)), phi_p_BPo[i],
          rotation_generator.phi(), b_p_BPo[i].cast<symbolic::Expression>(),
          R_BP_return.B_[i][j].cast<symbolic::Expression>(),
          solvers::IntervalBinning::kLogarithmic);
    }
  }

  std::array<Eigen::Vector3d, 6> p_PBi;
  std::array<Eigen::Vector3d, 6> p_BAi;
  std::array<double, 6> leg_length;

  p_BAi[0] << 0, 0, 0;
  p_PBi[0] << 0, 0, 0;
  leg_length[0] = 1;

  p_BAi[1] << 1.107915, 0, 0;
  p_PBi[1] << 0.542805, 0, 0;
  leg_length[1] = 0.645275;

  p_BAi[2] << 0.549094, 0.756063, 0;
  p_PBi[2] << 0.956919, -0.528915, 0;
  leg_length[2] = 1.086284;

  p_BAi[3] << 0.735077, -0.223935, 0.525991;
  p_PBi[3] << 0.665885, -0.353482, 1.402538;
  leg_length[3] = 1.503439;

  p_BAi[4] << 0.514188, -0.526063, -0.368418;
  p_PBi[4] << 0.478359, 1.158742, 0.107672;
  leg_length[4] = 1.281933;

  p_BAi[5] << 0.590473, 0.094733, -0.205018;
  p_PBi[5] << -0.137087, -0.235121, 0.353913;
  leg_length[5] = 0.771071;
  for (int i = 0; i < 6; ++i) {
    AddLeg(p_PBi[i], p_BAi[i], leg_length[i], R_BP, p_BPo, phi_p_BPo,
           lambda_p_BPo, p_BPo_times_R_BP, use_linear_constraint, &prog);
  }

  if (use_linear_constraint) {
    assert(prog.lorentz_cone_constraints().empty());
    assert(prog.rotated_lorentz_cone_constraints().empty());
  }

  solvers::GurobiSolver solver;
  prog.SetSolverOption(solvers::GurobiSolver::id(), "OutputFlag", 1);
  prog.SetSolverOption(solvers::GurobiSolver::id(), "PoolSearchMode", 2);
  const int multiple_sol_count = 150;
  prog.SetSolverOption(solvers::GurobiSolver::id(), "PoolSolutions",
                       multiple_sol_count);
  const auto result = solver.Solve(prog);
  std::cout << result << "\n";

  const std::string position_file_name("stewart_platform_position.txt");
  RemoveFileIfExist(position_file_name);
  const std::string orientation_file_name("stewart_platform_orientation.txt");
  RemoveFileIfExist(orientation_file_name);
  const std::string result_file_name("stewart_platform_result.txt");
  RemoveFileIfExist(result_file_name);

  for (int count = 0; count < multiple_sol_count; ++count) {
    const Eigen::Matrix3d R_BP_sol = prog.GetSuboptimalSolution(R_BP, count);
    const Eigen::Matrix3d R_BP_proj =
        math::RotationMatrix<double>::ProjectToRotationMatrix(R_BP_sol)
            .matrix();
    const Eigen::Vector3d p_BPo_sol = prog.GetSuboptimalSolution(p_BPo, count);
    std::cout << "\np_BPo_sol : " << p_BPo_sol.transpose() << "\n";
    WriteRowVectorToFile(p_BPo_sol.transpose(), position_file_name);
    for (int i = 0; i < 3; ++i) {
      WriteRowVectorToFile(R_BP_proj.row(i), orientation_file_name);
    }
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector3d p_BBi = p_BPo_sol + R_BP_proj * p_PBi[i];
      const double leg_length_sol = (p_BBi - p_BAi[i]).norm();
      std::cout << "leg " << i << " length_sol: " << leg_length_sol
                << " length " << leg_length[i] << "\n";
    }
    const Eigen::Vector3d p_A1B1_B =
        p_BPo_sol + R_BP_proj * p_PBi[0] - p_BAi[0];
    const Eigen::Vector3d n1 = p_A1B1_B.normalized();
    Eigen::VectorXd sol(12);
    sol << n1, R_BP_proj.col(0), R_BP_proj.col(1), R_BP_proj.col(2);
    WriteRowVectorToFile(sol.transpose(), result_file_name);
  }

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
