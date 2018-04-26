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

  MixedIntegerLinearProgramLCP milp_lcp(q, M, Eigen::VectorXd::Constant(73, 2),
                                        Eigen::VectorXd::Constant(73, 2));

  GurobiSolver solver;
  // milp_lcp.get_mutable_prog()->SetSolverOption(GurobiSolver::id(),
  //                                             "FeasibilityTol", 1E-9);
  milp_lcp.get_mutable_prog()->SetSolverOption(GurobiSolver::id(),
                                               "PoolSearchMode", 2);
  milp_lcp.get_mutable_prog()->SetSolverOption(GurobiSolver::id(),
                                               "PoolSolutions", 1000);
  const auto result = solver.Solve(*(milp_lcp.get_mutable_prog()));

  std::cout << result << "\n";
  if (result == SolutionResult::kSolutionFound) {
    std::cout << "number of solutions: "
              << milp_lcp.prog().multiple_solutions_.size() << "\n";
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> polished_solution;
    for (int i = 0;
         i < static_cast<int>(milp_lcp.prog().multiple_solutions_.size());
         ++i) {
      const Eigen::VectorXd w_sol = milp_lcp.prog().multiple_solutions_[i].head(q.rows());
      const Eigen::VectorXd z_sol = milp_lcp.prog().multiple_solutions_[i].segment(q.rows(), q.rows());
      const Eigen::VectorXd b_sol = milp_lcp.prog().multiple_solutions_[i].tail(q.rows());
      //std::cout << "w:\n" << w_sol.transpose() << "\n";
      //std::cout << "z:\n" << z_sol.transpose() << "\n";
      //std::cout << "b:\n" << b_sol.transpose() << "\n";
      //std::cout << "w.*z:\n"
      //          << (w_sol.array() * z_sol.array()).transpose() << "\n";
      std::cout << "w.min: " << w_sol.minCoeff()
                << "\nz.min: " << z_sol.minCoeff() << "\n";

      Eigen::VectorXd w_polish, z_polish;
      const bool polished = milp_lcp.PolishSolution(b_sol, &w_polish, &z_polish);
      //std::cout << "w:\n" << w_polish.transpose() << "\n";
      //std::cout << "z:\n" << z_polish.transpose() << "\n";
      //std::cout << "w.*z:\n"
      //          << (w_polish.array() * z_polish.array()).transpose() << "\n";
      std::cout << "w.min: " << w_polish.minCoeff()
                << "\nz.min: " << z_polish.minCoeff() << "\n";
      if (polished) {
        polished_solution.push_back(std::make_tuple(w_polish, z_polish, b_sol));
      }
    }

    std::cout << "\nNumber of polished solution: " << polished_solution.size() << "\n";
    for (const auto& wzb_polished : polished_solution) {
      std::cout << "w_polish: " << std::get<0>(wzb_polished).transpose() << "\n";
      std::cout << "z_polish: " << std::get<1>(wzb_polished).transpose() << "\n";
      std::cout << "b_polish: " << std::get<2>(wzb_polished).transpose() << "\n";
    }
  }

  return 0;
}
}
}  // namespace solvers
}  // namespace drake

int main() { return drake::solvers::DoMain(); }
