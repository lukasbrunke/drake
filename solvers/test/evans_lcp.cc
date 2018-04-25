#include "drake/solvers/mixed_integer_linear_program_LCP.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "drake/solvers/gurobi_solver.h"

namespace drake {
namespace solvers {
namespace {
void ReadData(Eigen::VectorXd* q, Eigen::MatrixXd* M) {
  std::ifstream q_file("/home/hongkai/drake-distro/solvers/test/lcp_q.mat",
                       std::ios::in);
  std::vector<double> q_vector;
  if (!q_file.is_open()) {
    std::cerr << "Cannot open q file.\n";
  }
  double qi = 0;
  while (q_file >> qi) {
    q_vector.push_back(qi);
  }
  *q = Eigen::Map<Eigen::VectorXd>(q_vector.data(), q_vector.size());

  std::ifstream M_file("/home/hongkai/drake-distro/solvers/test/lcp_M.mat",
                       std::ios::in);
  if (!M_file.is_open()) {
    std::cerr << "Cannot open M file.\n";
  }
  M->resize(q->rows(), q->rows());
  std::string M_row;
  int row_count = 0;
  while (std::getline(M_file, M_row)) {
    int col_count = 0;
    std::istringstream iss_M_row(M_row);
    while (col_count <= q->rows()) {
      std::string M_ij;
      iss_M_row >> M_ij;
      std::istringstream(M_ij) >> (*M)(row_count, col_count);
      ++col_count;
    }
    ++row_count;
  }
}

int DoMain() {
  Eigen::VectorXd q;
  Eigen::MatrixXd M;
  ReadData(&q, &M);

  MixedIntegerLinearProgramLCP milp_lcp(q, M, Eigen::VectorXd::Constant(73, 2), Eigen::VectorXd::Constant(73, 2));

  GurobiSolver solver;
  milp_lcp.get_mutable_prog()->SetSolverOption(GurobiSolver::id(), "FeasibilityTol", 1E-9);
  const auto result = solver.Solve(*(milp_lcp.get_mutable_prog()));

  std::cout << result << "\n";
  if (result == SolutionResult::kSolutionFound) {
    const auto w_sol = milp_lcp.prog().GetSolution(milp_lcp.w());
    const auto z_sol = milp_lcp.prog().GetSolution(milp_lcp.z());
    const auto b_sol = milp_lcp.prog().GetSolution(milp_lcp.b());
    std::cout << "w:\n" << w_sol.transpose() << "\n";
    std::cout << "z:\n" << z_sol.transpose() << "\n";
    std::cout << "b:\n" << b_sol.transpose() << "\n";
    std::cout << "w.*z:\n" << (w_sol.array() * z_sol.array()).transpose() << "\n";
    std::cout << "w.min: " << w_sol.minCoeff() << "\nz.min: " << z_sol.minCoeff() << "\n";

    Eigen::VectorXd w_polish, z_polish;
    milp_lcp.PolishSolution(b_sol, &w_polish, &z_polish);
    std::cout << "w:\n" << w_polish.transpose() << "\n";
    std::cout << "z:\n" << z_polish.transpose() << "\n";
    std::cout << "w.min: " << w_polish.minCoeff() << "\nz.min: " << z_polish.minCoeff() << "\n";
    std::cout << "w.*z:\n" << (w_polish.array() * z_polish.array()).transpose() << "\n";
  }
  return 0;
}
}
}  // namespace solvers
}  // namespace drake

int main() { return drake::solvers::DoMain(); }
