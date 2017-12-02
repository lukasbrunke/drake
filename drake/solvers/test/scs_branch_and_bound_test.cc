#include "drake/solvers/scs_branch_and_bound.h"

#include <Eigen/SparseCore>
#include <gtest/gtest.h>

namespace drake {
namespace solvers {
/**
 * This class exposes all the protected and private members of
 * ScsBranchAndBound, so that we can test its internal implementation.
 */
class ScsBranchAndBoundTest {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ScsBranchAndBoundTest)

  ScsBranchAndBoundTest(const SCS_PROBLEM_DATA& scs_data, const SCS_CONE& cone,
                        double cost_constant,
                        const std::list<int>& binary_var_indices)
      : bnb_tree_{std::make_unique<ScsBranchAndBound>(
            scs_data, cone, cost_constant, binary_var_indices)} {}

  ScsBranchAndBound* bnb_tree() const { return bnb_tree_.get(); }

  double best_upper_bound() const { return bnb_tree_->best_upper_bound_; }

  double best_lower_bound() const { return bnb_tree_->best_lower_bound_; }

  const std::list<ScsNode*>& active_leaves() const {
    return bnb_tree_->active_leaves_;
  }

  SCS_CONE* cone() const { return bnb_tree_->cone_.get(); }

  SCS_SETTINGS scs_settings() const { return bnb_tree_->settings_; }

  ScsNode* root() const { return bnb_tree_->root_.get(); }

  int PickBranchingVariable(const ScsNode& node) const {
    return bnb_tree_->PickBranchingVariable(node);
  }

  ScsNode* PickBranchingNode() const { return bnb_tree_->PickBranchingNode(); }

  scs_int SolveRootNode() { return bnb_tree_->SolveRootNode(); }

  bool IsNodeFathomed(const ScsNode& node) {
    return bnb_tree_->IsNodeFathomed(node);
  }

  bool IsConverged() const { return bnb_tree_->IsConverged(); }

  void BranchAndSolve(ScsNode* node, int branch_var_index) {
    return bnb_tree_->BranchAndSolve(node, branch_var_index);
  }

  std::vector<scs_float> RecoverSolutionFromNode(const ScsNode& node, const scs_float* const sol_x) const {
    std::vector<scs_float> mip_sol_x;
    bnb_tree_->RecoverSolutionFromNode(node, sol_x, &mip_sol_x);
    return mip_sol_x;
  }

 private:
  std::unique_ptr<ScsBranchAndBound> bnb_tree_;
};

namespace {
std::unique_ptr<AMatrix, void (*)(AMatrix*)> ConstructScsAmatrix(
    const Eigen::SparseMatrix<double>& A) {
  AMatrix* scs_A = static_cast<AMatrix*>(malloc(sizeof(AMatrix)));
  scs_A->m = A.rows();
  scs_A->n = A.cols();
  scs_A->x =
      static_cast<scs_float*>(scs_calloc(A.nonZeros(), sizeof(scs_float)));
  scs_A->i = static_cast<scs_int*>(scs_calloc(A.nonZeros(), sizeof(scs_int)));
  scs_A->p = static_cast<scs_int*>(scs_calloc(scs_A->n + 1, sizeof(scs_int)));
  for (int i = 0; i < A.nonZeros(); ++i) {
    scs_A->x[i] = *(A.valuePtr() + i);
    scs_A->i[i] = *(A.innerIndexPtr() + i);
  }
  for (int i = 0; i < scs_A->n + 1; ++i) {
    scs_A->p[i] = *(A.outerIndexPtr() + i);
  }
  return std::unique_ptr<AMatrix, void (*)(AMatrix*)>(scs_A, &freeAMatrix);
}

Eigen::SparseMatrix<double> ScsAmatrixToEigenSparseMatrix(
    const AMatrix& scs_A) {
  Eigen::SparseMatrix<double> A(scs_A.m, scs_A.n);
  A.reserve(scs_A.p[scs_A.n]);
  A.setZero();
  for (int j = 0; j < scs_A.n; ++j) {
    for (int i = scs_A.p[j]; i < scs_A.p[j + 1]; ++i) {
      A.insert(scs_A.i[i], j) = scs_A.x[i];
    }
  }
  A.makeCompressed();
  return A;
}

void IsAmatrixEqual(const AMatrix& A1, const AMatrix& A2, double tol) {
  EXPECT_EQ(A1.m, A2.m);
  EXPECT_EQ(A1.n, A2.n);
  for (int i = 0; i < A1.n + 1; ++i) {
    EXPECT_EQ(A1.p[i], A2.p[i]);
  }
  for (int i = 0; i < A1.p[A1.n]; ++i) {
    EXPECT_NEAR(A1.x[i], A2.x[i], tol);
    EXPECT_EQ(A1.i[i], A2.i[i]);
  }
}

/**
 * Test if two lists are equal, namely list1[i] = list2[i]
 */
template <typename T>
bool IsListEqual(const std::list<T>& list1, const std::list<T>& list2) {
  if (list1.size() != list2.size()) {
    return false;
  }
  auto it2 = list2.begin();
  for (auto it1 = list1.begin(); it1 != list1.end(); ++it1) {
    if (*it1 != *it2) {
      return false;
    }
    ++it2;
  }
  return true;
}

template <typename T>
bool IsListEqualAfterReshuffle(const std::list<T>& list1,
                               const std::list<T>& list2) {
  if (list1.size() != list2.size()) {
    return false;
  }
  std::list<T> list1_remaining = list1;
  std::list<T> list2_remaining = list2;
  auto it1 = list1_remaining.begin();
  while (!list1_remaining.empty()) {
    bool found_match = false;
    for (auto it2 = list2_remaining.begin(); it2 != list2_remaining.end();
         ++it2) {
      if (*it1 == *it2) {
        list1_remaining.erase(it1);
        list2_remaining.erase(it2);
        it1 = list1_remaining.begin();
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      return false;
    }
  }
  return true;
}

GTEST_TEST(TestIsListEqualAfterReshuffle, test) {
  EXPECT_TRUE(IsListEqualAfterReshuffle<int>({}, {}));
  EXPECT_TRUE(IsListEqualAfterReshuffle<int>({1}, {1}));
  EXPECT_TRUE(IsListEqualAfterReshuffle<int>({1, 2}, {2, 1}));
  EXPECT_TRUE(IsListEqualAfterReshuffle<int>({1, 2, 1}, {1, 2, 1}));
  EXPECT_TRUE(IsListEqualAfterReshuffle<int>({1, 1, 2}, {1, 2, 1}));

  EXPECT_FALSE(IsListEqualAfterReshuffle<int>({}, {1}));
  EXPECT_FALSE(IsListEqualAfterReshuffle<int>({1, 2}, {1}));
  EXPECT_FALSE(IsListEqualAfterReshuffle<int>({1, 1}, {1, 2}));
}

// Determine if the two constraints
// A1*x+s = b1
// and
// A2*x+s = b2
// are the same constraints.
// A1,A2,b1 and b2 are obtained by relaxing the constraint
// A*x + s = b
// y ∈ {0, 1}
void IsSameRelaxedConstraint(const AMatrix& A1, const AMatrix& A2,
                             const scs_float* const b1, const scs_float* b2,
                             double tol) {
  EXPECT_EQ(A1.m, A2.m);
  EXPECT_EQ(A1.n, A2.n);
  for (int i = 0; i < A1.n + 1; ++i) {
    EXPECT_EQ(A1.p[i], A2.p[i]);
  }
  for (int i = 0; i < A1.p[A1.n]; ++i) {
    EXPECT_EQ(A1.i[i], A2.i[i]);
    EXPECT_NEAR(A1.x[i], A2.x[i], tol);
  }
  for (int i = 0; i < A1.m; ++i) {
    EXPECT_NEAR(b1[i], b2[i], tol);
  }
}

GTEST_TEST(TestSparseMatrix, TestConversion) {
  // Test if ScsAmatrixToEigenSparseMatrix is the inversion of
  // ConstructScsAmatrix
  std::vector<Eigen::SparseMatrix<double>> X;
  Eigen::SparseMatrix<double> Xi(2, 2);
  Xi.setZero();
  X.push_back(Xi);
  Xi.setIdentity();
  X.push_back(Xi);
  Xi.setZero();
  Xi.insert(0, 0) = 1;
  X.push_back(Xi);
  Xi.setZero();
  Xi.insert(1, 0) = 2;
  Xi.insert(0, 1) = 3;
  X.push_back(Xi);
  for (const auto& Xi : X) {
    EXPECT_TRUE(Xi.isApprox(
        ScsAmatrixToEigenSparseMatrix(*ConstructScsAmatrix(Xi)), 1E-10));
  }
}

void freeCone(SCS_CONE* cone) {
  if (cone) {
    if (cone->q) {
      scs_free(cone->q);
    }
    if (cone->s) {
      scs_free(cone->s);
    }
    if (cone->p) {
      scs_free(cone->p);
    }
    scs_free(cone);
  }
}

std::unique_ptr<SCS_CONE, void (*)(SCS_CONE*)> DeepCopyScsCone(
    const SCS_CONE* const cone) {
  SCS_CONE* new_cone = static_cast<SCS_CONE*>(scs_calloc(1, sizeof(SCS_CONE)));
  new_cone->f = cone->f;
  new_cone->l = cone->l;
  new_cone->qsize = cone->qsize;
  if (cone->q) {
    new_cone->q =
        static_cast<scs_int*>(scs_calloc(new_cone->qsize, sizeof(scs_int)));
    for (int i = 0; i < new_cone->qsize; ++i) {
      new_cone->q[i] = cone->q[i];
    }
  } else {
    new_cone->q = nullptr;
  }
  new_cone->ssize = cone->ssize;
  if (cone->s) {
    new_cone->s =
        static_cast<scs_int*>(scs_calloc(new_cone->ssize, sizeof(scs_int)));
    for (int i = 0; i < new_cone->ssize; ++i) {
      new_cone->s[i] = cone->s[i];
    }
  } else {
    new_cone->s = nullptr;
  }
  new_cone->ep = cone->ep;
  new_cone->ed = cone->ed;
  new_cone->psize = cone->psize;
  if (cone->p) {
    new_cone->p =
        static_cast<scs_float*>(scs_calloc(new_cone->psize, sizeof(scs_float)));
    for (int i = 0; i < new_cone->psize; ++i) {
      new_cone->p[i] = cone->p[i];
    }
  } else {
    new_cone->p = nullptr;
  }
  return std::unique_ptr<SCS_CONE, void (*)(SCS_CONE*)>(new_cone, &freeCone);
}

void IsConeEqual(const SCS_CONE& cone1, const SCS_CONE& cone2) {
  EXPECT_EQ(cone1.f, cone2.f);
  EXPECT_EQ(cone1.l, cone2.l);
  EXPECT_EQ(cone1.qsize, cone2.qsize);
  if (cone1.q) {
    EXPECT_NE(cone2.q, nullptr);
    for (int i = 0; i < cone1.qsize; ++i) {
      EXPECT_EQ(cone1.q[i], cone2.q[i]);
    }
  } else {
    EXPECT_EQ(cone2.q, nullptr);
  }
  EXPECT_EQ(cone1.ssize, cone2.ssize);
  if (cone1.s) {
    EXPECT_NE(cone2.s, nullptr);
    for (int i = 0; i < cone1.ssize; ++i) {
      EXPECT_EQ(cone1.s[i], cone2.s[i]);
    }
  } else {
    EXPECT_EQ(cone2.s, nullptr);
  }
  EXPECT_EQ(cone1.ep, cone2.ep);
  EXPECT_EQ(cone1.ed, cone2.ed);
  EXPECT_EQ(cone1.psize, cone2.psize);
  if (cone1.p) {
    EXPECT_NE(cone2.s, nullptr);
    for (int i = 0; i < cone1.psize; ++i) {
      EXPECT_EQ(cone1.p[i], cone2.p[i]);
    }
  } else {
    EXPECT_EQ(cone2.p, nullptr);
  }
}

void free_scs_pointer(void* scs_pointer) { scs_free(scs_pointer); }

// Store the data for a mixed-inteter optimization problem
// min cᵀx + d
// s.t Ax + s = b
//     s in cone
//     x(binary_var_indices_) are binary variables
struct MIPdata {
  MIPdata(const Eigen::SparseMatrix<double>& A, const scs_float* const b,
          const scs_float* const c, double d, const SCS_CONE& cone,
          const std::list<int>& binary_var_indices)
      : A_{ConstructScsAmatrix(A)},
        b_{static_cast<scs_float*>(scs_calloc(A_->m, sizeof(scs_float))),
           &free_scs_pointer},
        c_{static_cast<scs_float*>(scs_calloc(A_->n, sizeof(scs_float))),
           &free_scs_pointer},
        d_{d},
        cone_{DeepCopyScsCone(&cone)},
        binary_var_indices_{binary_var_indices} {
    for (int i = 0; i < A_->m; ++i) {
      b_.get()[i] = b[i];
    }
    for (int i = 0; i < A_->n; ++i) {
      c_.get()[i] = c[i];
    }
  }
  std::unique_ptr<AMatrix, void (*)(AMatrix*)> A_;
  std::unique_ptr<scs_float, void (*)(void*)> b_;
  std::unique_ptr<scs_float, void (*)(void*)> c_;
  double d_;
  std::unique_ptr<SCS_CONE, void (*)(SCS_CONE*)> cone_;
  std::list<int> binary_var_indices_;
};

// Construct the problem data for the mixed-integer linear program
// min x(0) + 2x(1) -3x(3) + 1
// s.t x(0) + x(1) + 2x(3) = 2
//     x(1) - 3.1 x(2) >= 1
//     x(2) + 1.2x(3) - x(0) <= 5
//     x(0), x(2) are binary
// At the root node, the optimizer should obtain an integral solution. So the
// root node does not need to branch.
// The optimal solution is (0, 1, 0, 0.5), the optimal cost is 1.5
MIPdata ConstructMILPExample1() {
  Eigen::Matrix<double, 3, 4> A;
  // clang-format off
  A << 1, 1, 0, 2,
       0, -1, 3.1, 0,
       -1, 0, 1, 1.2;
  // clang-format on
  const scs_float b[3] = {2, -1, 5};
  const scs_float c[4] = {1, 2, 0, -3};
  SCS_CONE cone;
  cone.f = 1;
  cone.l = 2;
  cone.q = nullptr;
  cone.qsize = 0;
  cone.s = nullptr;
  cone.ssize = 0;
  cone.ed = 0;
  cone.ep = 0;
  cone.p = nullptr;
  cone.psize = 0;
  return MIPdata(A.sparseView(), b, c, 1, cone, {0, 2});
}

// Construct the problem data for the mixed-integer linear program
// min x₀ + 2x₁ - 3x₂ - 4x₃ + 4.5x₄ + 1
// s.t 2x₀ + x₂ + 1.5x₃ + x₄ = 4.5
//     1 ≤ 2x₀ + 4x₃ + x₄ ≤ 7
//     -2 ≤ 3x₁ + 2x₂ - 5x₃ + x₄ ≤ 7
//     -5 ≤ x₁ + x₂ + 2x₃ ≤ 10
//     -10 ≤ x₁ ≤ 10
//     x₀, x₂, x₄ are binary variables.
// The optimal solution is (1, 1/3, 1, 1, 0), with optimal cost -13/3.
MIPdata ConstructMILPExample2() {
  Eigen::Matrix<double, 9, 5> A;
  // clang-format off
  A << 2, 0, 1, 1.5, 1,
      2, 0, 0, 4, 1,
      -2, 0, 0, -4, -1,
      0, 3, 2, -5, 1,
      0, -3, -2, 5, -1,
      0, 1, 1, 2, 0,
      0, -1, -1, -2, 0,
      0, 1, 0, 0, 0,
      0, -1, 0, 0, 0;
  // clang-format on
  const scs_float b[9] = {4.5, 7, -1, 7, 2, 10, 5, 10, 10};
  const scs_float c[5] = {1, 2, -3, -4, 4.5};
  SCS_CONE cone;
  cone.f = 1;
  cone.l = 8;
  cone.q = nullptr;
  cone.qsize = 0;
  cone.s = nullptr;
  cone.ssize = 0;
  cone.ed = 0;
  cone.ep = 0;
  cone.p = nullptr;
  cone.psize = 0;
  return MIPdata(A.sparseView(), b, c, 1, cone, {0, 2, 4});
}

// A mixed-integer optimization problem that is un-bounded.
// min x0 + 2*x1 + 3*x2 + 2.5*x3 + 2
// s.t     x0 + x1 - x2 + x3 <= 3
//     1<= x0 + 2*x1 -2*x2 + 4*x3 <= 3
//     x0, x2 are binary
MIPdata ConstructMILPExample3() {
  Eigen::Matrix<double, 3, 4> A;
  // clang-format off
  A << 1, 1, -1, 1,
       1, 2, -2, 4,
       -1, -2, 2, -4;
  // clang-format on
  const scs_float b[3] = {3, 3, -1};
  const scs_float c[4] = {1, 2, 3, 2.5};
  SCS_CONE cone;
  cone.f = 0;
  cone.l = 3;
  cone.q = nullptr;
  cone.qsize = 0;
  cone.s = nullptr;
  cone.ssize = 0;
  cone.ed = 0;
  cone.ep = 0;
  cone.p = nullptr;
  cone.psize = 0;
  return MIPdata(A.sparseView(), b, c, 2, cone, {0, 2});
}

std::unique_ptr<ScsNode> ConstructMILPExample1RootNode() {
  MIPdata mip_data = ConstructMILPExample1();
  return ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                    mip_data.c_.get(), *(mip_data.cone_),
                                    mip_data.binary_var_indices_, mip_data.d_);
}

std::unique_ptr<ScsNode> ConstructMILPExample2RootNode() {
  MIPdata mip_data = ConstructMILPExample2();
  return ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                    mip_data.c_.get(), *(mip_data.cone_),
                                    mip_data.binary_var_indices_, mip_data.d_);
}

std::unique_ptr<ScsNode> ConstructMILPExample3RootNode() {
  MIPdata mip_data = ConstructMILPExample3();
  return ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                    mip_data.c_.get(), *(mip_data.cone_),
                                    mip_data.binary_var_indices_, mip_data.d_);
}

void SetScsSettingToDefault(SCS_SETTINGS* settings) {
  settings->alpha = ALPHA;
  settings->cg_rate = CG_RATE;
  settings->eps = EPS;
  settings->max_iters = MAX_ITERS;
  settings->normalize = NORMALIZE;
  settings->rho_x = RHO_X;
  settings->scale = SCALE;
  settings->verbose = VERBOSE;
  settings->warm_start = WARM_START;
}

GTEST_TEST(TestScsNode, TestConstructor) {
  ScsNode node(2, 3);
  EXPECT_EQ(node.A()->m, 2);
  EXPECT_EQ(node.A()->n, 3);
  EXPECT_EQ(node.y_index(), -1);
  EXPECT_EQ(node.y_val(), -1);
  EXPECT_TRUE(std::isnan(node.cost()));
  EXPECT_EQ(node.cost_constant(), 0);
  EXPECT_FALSE(node.found_integral_sol());
  EXPECT_TRUE(node.binary_var_indices().empty());
  EXPECT_EQ(node.left_child(), nullptr);
  EXPECT_EQ(node.right_child(), nullptr);
  EXPECT_EQ(node.parent(), nullptr);
}

void TestConstructScsRootNode(const AMatrix& A, const scs_float* const b,
                              const scs_float* const c, const SCS_CONE& cone,
                              const std::list<int>& binary_var_indices,
                              double cost_constant) {
  const auto root = ScsNode::ConstructRootNode(
      A, b, c, cone, binary_var_indices, cost_constant);
  EXPECT_EQ(root->y_index(), -1);
  EXPECT_EQ(root->left_child(), nullptr);
  EXPECT_EQ(root->right_child(), nullptr);
  EXPECT_EQ(root->parent(), nullptr);
  const Eigen::SparseMatrix<double> A_sparse = ScsAmatrixToEigenSparseMatrix(A);
  std::vector<Eigen::Triplet<double>> root_A_triplets;
  for (int i = 0; i < A_sparse.rows(); ++i) {
    for (int j = 0; j < A_sparse.cols(); ++j) {
      if (A_sparse.coeff(i, j) != 0) {
        root_A_triplets.emplace_back(
            i + (i >= cone.f ? 2 * binary_var_indices.size() : 0), j,
            A_sparse.coeff(i, j));
      }
    }
  }
  int binary_var_count = 0;
  for (auto it = binary_var_indices.begin(); it != binary_var_indices.end();
       ++it) {
    root_A_triplets.emplace_back(cone.f + binary_var_count * 2, *it, -1);
    root_A_triplets.emplace_back(cone.f + binary_var_count * 2 + 1, *it, 1);
    ++binary_var_count;
  }
  Eigen::SparseMatrix<double> root_A(A.m + 2 * binary_var_indices.size(), A.n);
  root_A.setFromTriplets(root_A_triplets.begin(), root_A_triplets.end());
  scs_float* root_b = new scs_float[A.m + 2 * binary_var_indices.size()];
  for (int i = 0; i < cone.f; ++i) {
    root_b[i] = b[i];
  }
  for (int i = 0; i < static_cast<int>(binary_var_indices.size()); ++i) {
    root_b[cone.f + 2 * i] = 0;
    root_b[cone.f + 1 + 2 * i] = 1;
  }
  for (int i = cone.f; i < A.m; ++i) {
    root_b[2 * binary_var_indices.size() + i] = b[i];
  }
  auto root_scs_A = ConstructScsAmatrix(root_A);

  IsSameRelaxedConstraint(*root_scs_A, *(root->A()), root_b, root->b(), 0);

  for (int i = 0; i < root_A.rows(); ++i) {
    EXPECT_EQ(root_b[i], root->b()[i]);
  }
  delete[] root_b;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(c[i], root->c()[i]);
  }
  EXPECT_EQ(root->cost_constant(), cost_constant);
  // Check the cones
  auto root_cone_expected = DeepCopyScsCone(&cone);
  root_cone_expected->l += 2 * binary_var_indices.size();
  IsConeEqual(*(root->cone()), *root_cone_expected);

  EXPECT_FALSE(root->found_integral_sol());
  IsListEqual(root->binary_var_indices(), binary_var_indices);

  EXPECT_TRUE(root->IsLeaf());
}

GTEST_TEST(TestScsNode, TestConstructRoot1) {
  MIPdata mip_data = ConstructMILPExample1();
  TestConstructScsRootNode(*(mip_data.A_), mip_data.b_.get(), mip_data.c_.get(),
                           *(mip_data.cone_), mip_data.binary_var_indices_,
                           mip_data.d_);
}

GTEST_TEST(TestScsNode, TestConstructRoot2) {
  MIPdata mip_data = ConstructMILPExample2();
  TestConstructScsRootNode(*(mip_data.A_), mip_data.b_.get(), mip_data.c_.get(),
                           *(mip_data.cone_), mip_data.binary_var_indices_,
                           mip_data.d_);
}

GTEST_TEST(TestScsNode, TestConstructRoot3) {
  MIPdata mip_data = ConstructMILPExample3();
  TestConstructScsRootNode(*(mip_data.A_), mip_data.b_.get(), mip_data.c_.get(),
                           *(mip_data.cone_), mip_data.binary_var_indices_,
                           mip_data.d_);
}

GTEST_TEST(TestScsNode, TestConstructRootError) {
  MIPdata mip_data = ConstructMILPExample1();
  EXPECT_THROW(ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                          mip_data.c_.get(), *(mip_data.cone_),
                                          {0, 4}, mip_data.d_),
               std::runtime_error);
  EXPECT_THROW(ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                          mip_data.c_.get(), *(mip_data.cone_),
                                          {-1, 0}, mip_data.d_),
               std::runtime_error);
  EXPECT_THROW(ScsNode::ConstructRootNode(*(mip_data.A_), mip_data.b_.get(),
                                          mip_data.c_.get(), *(mip_data.cone_),
                                          {1, 1, 4}, mip_data.d_),
               std::runtime_error);
}

void TestBranching(ScsNode* root, int branch_var_index,
                   const std::list<int>& binary_var_indices_child,
                   const scs_float* const b_left,
                   const scs_float* const b_right, const scs_float* c_child,
                   double cost_constant_left, double cost_constant_right,
                   const AMatrix& A_child) {
  root->Branch(branch_var_index);
  EXPECT_NE(root->left_child(), nullptr);
  EXPECT_NE(root->right_child(), nullptr);
  EXPECT_EQ(root->left_child()->parent(), root);
  EXPECT_EQ(root->right_child()->parent(), root);

  EXPECT_FALSE(root->IsLeaf());
  EXPECT_TRUE(root->left_child()->IsLeaf());
  EXPECT_TRUE(root->right_child()->IsLeaf());

  IsListEqual(root->left_child()->binary_var_indices(),
              binary_var_indices_child);
  IsListEqual(root->right_child()->binary_var_indices(),
              binary_var_indices_child);

  EXPECT_EQ(root->left_child()->A()->m, root->A()->m - 2);
  EXPECT_EQ(root->right_child()->A()->m, root->A()->m - 2);
  EXPECT_EQ(root->left_child()->A()->n, root->A()->n - 1);
  EXPECT_EQ(root->right_child()->A()->n, root->A()->n - 1);

  const double tol{1E-10};
  for (int i = 0; i < root->A()->n - 1; ++i) {
    EXPECT_NEAR(root->left_child()->c()[i], c_child[i], tol);
    EXPECT_NEAR(root->right_child()->c()[i], c_child[i], tol);
  }

  EXPECT_NEAR(root->left_child()->cost_constant(), cost_constant_left, tol);
  EXPECT_NEAR(root->right_child()->cost_constant(), cost_constant_right, tol);

  for (int i = 0; i < root->A()->m - 2; ++i) {
    EXPECT_EQ(root->left_child()->b()[i], b_left[i]);
    EXPECT_EQ(root->right_child()->b()[i], b_right[i]);
  }

  IsAmatrixEqual(A_child, *(root->left_child()->A()), tol);
  IsAmatrixEqual(A_child, *(root->right_child()->A()), tol);

  // Check if the y_index and y_val are correct in the child nodes.
  EXPECT_EQ(root->left_child()->y_index(), branch_var_index);
  EXPECT_EQ(root->right_child()->y_index(), branch_var_index);
  EXPECT_EQ(root->left_child()->y_val(), 0);
  EXPECT_EQ(root->right_child()->y_val(), 1);

  auto child_cone = DeepCopyScsCone(root->cone());
  child_cone->l -= 2;
  IsConeEqual(*child_cone, *(root->left_child()->cone()));
  IsConeEqual(*child_cone, *(root->right_child()->cone()));
}

GTEST_TEST(TestScsNode, TestBranch1) {
  const auto root = ConstructMILPExample1RootNode();

  // Branch on x0
  const scs_float b_left[5] = {2, 0, 1, -1, 5};
  const scs_float b_right[5] = {1, 0, 1, -1, 6};
  const scs_float c_child[3] = {2, 0, -3};
  // Check the left-hand side matrix A
  // A_child = [ 1    0    2]
  //           [ 0   -1    0]
  //           [ 0    1    0]
  //           [-1  3.1    0]
  //           [ 0    1  1.2]
  std::vector<Eigen::Triplet<double>> A_child_triplets;
  A_child_triplets.emplace_back(0, 0, 1);
  A_child_triplets.emplace_back(0, 2, 2);
  A_child_triplets.emplace_back(1, 1, -1);
  A_child_triplets.emplace_back(2, 1, 1);
  A_child_triplets.emplace_back(3, 0, -1);
  A_child_triplets.emplace_back(3, 1, 3.1);
  A_child_triplets.emplace_back(4, 1, 1);
  A_child_triplets.emplace_back(4, 2, 1.2);
  Eigen::SparseMatrix<double> A_child(5, 3);
  A_child.setFromTriplets(A_child_triplets.begin(), A_child_triplets.end());
  const auto scs_A_child = ConstructScsAmatrix(A_child);
  const std::list<int> binary_var_indices_child = {1};

  TestBranching(root.get(), 0, binary_var_indices_child, b_left, b_right,
                c_child, 1, 2, *scs_A_child);
}

GTEST_TEST(TestScsNode, TestBranch2) {
  const auto root = ConstructMILPExample1RootNode();

  // Branch on x2
  const scs_float b_left[5] = {2, 0, 1, -1, 5};
  const scs_float b_right[5] = {2, 0, 1, -4.1, 4};
  const scs_float c_child[3] = {1, 2, -3};
  Eigen::Matrix<double, 5, 3> A;
  // clang-format off
  A << 1, 1, 2,
      -1, 0, 0,
      1, 0, 0,
      0, -1, 0,
      -1, 0, 1.2;
  // clang-format on
  const auto scs_A_child = ConstructScsAmatrix(A.sparseView());
  const std::list<int> binary_var_indices_child = {0};

  TestBranching(root.get(), 2, binary_var_indices_child, b_left, b_right,
                c_child, 1, 1, *scs_A_child);
}

GTEST_TEST(TestScsNode, TestBranch3) {
  const auto root = ConstructMILPExample2RootNode();

  // Branch on x0
  const scs_float b_left[13] = {4.5, 0, 1, 0, 1, 7, -1, 7, 2, 10, 5, 10, 10};
  const scs_float b_right[13] = {2.5, 0, 1, 0, 1, 5, 1, 7, 2, 10, 5, 10, 10};
  const scs_float c_child[4] = {2, -3, -4, 4.5};
  Eigen::Matrix<double, 13, 4> A;
  // clang-format off
  A << 0, 1, 1.5, 1,
      0, -1, 0, 0,
      0, 1, 0, 0,
      0, 0, 0, -1,
      0, 0, 0, 1,
      0, 0, 4, 1,
      0, 0, -4, -1,
      3, 2, -5, 1,
      -3, -2, 5, -1,
      1, 1, 2, 0,
      -1, -1, -2, 0,
      1, 0, 0, 0,
      -1, 0, 0, 0;
  // clang-format on
  const auto scs_A_child = ConstructScsAmatrix(A.sparseView());
  const std::list<int> binary_var_indices_child = {1, 3};

  TestBranching(root.get(), 0, binary_var_indices_child, b_left, b_right,
                c_child, 1, 2, *scs_A_child);
}

GTEST_TEST(TestScsNode, TestBranch4) {
  const auto root = ConstructMILPExample2RootNode();

  // Branch on x2
  const scs_float b_left[13] = {4.5, 0, 1, 0, 1, 7, -1, 7, 2, 10, 5, 10, 10};
  const scs_float b_right[13] = {3.5, 0, 1, 0, 1, 7, -1, 5, 4, 9, 6, 10, 10};
  const scs_float c_child[4] = {1, 2, -4, 4.5};
  Eigen::Matrix<double, 13, 4> A;
  // clang-format off
  A << 2, 0, 1.5, 1,
      -1, 0, 0, 0,
      1, 0, 0, 0,
      0, 0, 0, -1,
      0, 0, 0, 1,
      2, 0, 4, 1,
      -2, 0, -4, -1,
      0, 3, -5, 1,
      0, -3, 5, -1,
      0, 1, 2, 0,
      0, -1, -2, 0,
      0, 1, 0, 0,
      0, -1, 0, 0;
  // clang-format on
  const auto scs_A_child = ConstructScsAmatrix(A.sparseView());
  const std::list<int> binary_var_indices_child = {0, 3};

  TestBranching(root.get(), 2, binary_var_indices_child, b_left, b_right,
                c_child, 1, -2, *scs_A_child);
}

GTEST_TEST(TestScsNode, TestBranch5) {
  const auto root = ConstructMILPExample2RootNode();

  // Branch on x4
  const scs_float b_left[13] = {4.5, 0, 1, 0, 1, 7, -1, 7, 2, 10, 5, 10, 10};
  const scs_float b_right[13] = {3.5, 0, 1, 0, 1, 6, 0, 6, 3, 10, 5, 10, 10};
  const scs_float c_child[4] = {1, 2, -3, -4};
  Eigen::Matrix<double, 13, 4> A;
  // clang-format off
  A << 2, 0, 1, 1.5,
      -1, 0, 0, 0,
      1, 0, 0, 0,
      0, 0, -1, 0,
      0, 0, 1, 0,
      2, 0, 0, 4,
      -2, 0, 0, -4,
      0, 3, 2, -5,
      0, -3, -2, 5,
      0, 1, 1, 2,
      0, -1, -1, -2,
      0, 1, 0, 0,
      0, -1, 0, 0;
  // clang-format on
  const auto scs_A_child = ConstructScsAmatrix(A.sparseView());
  const std::list<int> binary_var_indices_child = {0, 2};

  TestBranching(root.get(), 4, binary_var_indices_child, b_left, b_right,
                c_child, 1, 5.5, *scs_A_child);
}

GTEST_TEST(TestScsNode, TestBranchError) {
  const auto root = ConstructMILPExample1RootNode();

  // Branch on a variable that is NOT binary.
  EXPECT_THROW(root->Branch(1), std::runtime_error);
}

void TestNodeSolve(const ScsNode& node, scs_int scs_status_expected,
                   scs_float cost, const scs_float* const x,
                   bool found_integral_sol, double tol) {
  const scs_int scs_status = node.scs_info().statusVal;
  EXPECT_EQ(scs_status, scs_status_expected);

  if (scs_status == SCS_SOLVED || scs_status == SCS_SOLVED_INACCURATE) {
    EXPECT_NEAR(node.cost(), cost, tol);
    for (int i = 0; i < node.A()->n; ++i) {
      EXPECT_NEAR(x[i], node.scs_sol()->x[i], tol);
    }
    EXPECT_EQ(node.found_integral_sol(), found_integral_sol);
  } else if (scs_status == SCS_INFEASIBLE ||
             scs_status == SCS_INFEASIBLE_INACCURATE) {
    EXPECT_EQ(node.cost(), std::numeric_limits<double>::infinity());
    EXPECT_FALSE(node.found_integral_sol());
  } else if (scs_status == SCS_UNBOUNDED ||
             scs_status == SCS_UNBOUNDED_INACCURATE) {
    EXPECT_TRUE(std::isinf(node.cost()));
    EXPECT_EQ(node.cost(), -std::numeric_limits<double>::infinity());
  } else {
    throw std::runtime_error("SCS does not give a meaningful answer.");
  }
}

scs_int SolveNodeWithDefaultSettings(ScsNode* node) {
  SCS_SETTINGS settings;
  SetScsSettingToDefault(&settings);
  return node->Solve(settings);
}

GTEST_TEST(TestScsNode, TestSolve1) {
  // Solve the root node
  const auto root = ConstructMILPExample1RootNode();

  SolveNodeWithDefaultSettings(root.get());

  const scs_float x_expected[4] = {0, 1, 0, 0.5};
  TestNodeSolve(*root, SCS_SOLVED, 1.5, x_expected, true, 1E-3);
}

GTEST_TEST(TestScsNode, TestSolve2) {
  // Solve the root node
  const auto root = ConstructMILPExample2RootNode();

  SolveNodeWithDefaultSettings(root.get());

  const scs_float x_expected[5] = {0.7, 1, 1, 1.4, 0};
  TestNodeSolve(*root, SCS_SOLVED, -4.9, x_expected, false, 2E-2);
}

GTEST_TEST(TestScsNode, TestSolve3) {
  // Solve the root node
  const auto root = ConstructMILPExample3RootNode();

  SolveNodeWithDefaultSettings(root.get());
  TestNodeSolve(*root, SCS_UNBOUNDED, -std::numeric_limits<double>::infinity(),
                nullptr, false, 1E-3);
}

GTEST_TEST(TestScsNode, TestSolveChildNodes1) {
  // Solve the left and right child nodes of the root, by branching on x0.
  const auto root = ConstructMILPExample1RootNode();

  root->Branch(0);
  SolveNodeWithDefaultSettings(root->left_child());
  SolveNodeWithDefaultSettings(root->right_child());

  const scs_float x_expected_l[3] = {1, 0, 0.5};
  TestNodeSolve(*(root->left_child()), SCS_SOLVED, 1.5, x_expected_l, true,
                1E-3);

  const scs_float x_expected_r[3] = {1, 0, 0};
  TestNodeSolve(*(root->right_child()), SCS_SOLVED, 4, x_expected_r, true,
                2E-3);
}

GTEST_TEST(TestScsNode, TestSolveChildNodes2) {
  // Solve the left and right child nodes of the root, by branching on x2
  const auto root = ConstructMILPExample1RootNode();

  root->Branch(2);
  SolveNodeWithDefaultSettings(root->left_child());
  SolveNodeWithDefaultSettings(root->right_child());

  const scs_float x_expected_l[3] = {0, 1, 0.5};
  TestNodeSolve(*(root->left_child()), SCS_SOLVED, 1.5, x_expected_l, true,
                1E-3);

  const scs_float x_expected_r[3] = {0, 4.1, -1.05};
  TestNodeSolve(*(root->right_child()), SCS_SOLVED, 12.35, x_expected_r, true,
                4E-3);
}

GTEST_TEST(TestScsNode, TestSolveChildNodes3) {
  // Solve the left and right child nodes of the root, by branching on x0
  const auto root = ConstructMILPExample2RootNode();
  root->Branch(0);
  SolveNodeWithDefaultSettings(root->left_child());
  SolveNodeWithDefaultSettings(root->right_child());

  // The left node is infeasible.
  TestNodeSolve(*(root->left_child()), SCS_INFEASIBLE, NAN, nullptr, NAN, 1E-3);

  const scs_float x_expected_r[4] = {1.0 / 3.0, 1, 1, 0};
  TestNodeSolve(*(root->right_child()), SCS_SOLVED, -13.0 / 3.0, x_expected_r,
                true, 2E-2);
}

GTEST_TEST(TestScsNode, TestSolveChildNode4) {
  // Solve the left and right child nodes of the root, by branching on x2
  const auto root = ConstructMILPExample2RootNode();
  root->Branch(2);
  SolveNodeWithDefaultSettings(root->left_child());
  SolveNodeWithDefaultSettings(root->right_child());

  const scs_float x_expected_l[4] = {1, 2.0 / 3.0, 1, 1};
  TestNodeSolve(*(root->left_child()), SCS_SOLVED, 23.0 / 6.0, x_expected_l,
                true, 1E-2);

  const scs_float x_expected_r[4] = {0.7, 1, 1.4, 0};
  TestNodeSolve(*(root->right_child()), SCS_SOLVED, -4.9, x_expected_r, false,
                2E-2);
}

GTEST_TEST(TestScsNode, TestSolveChildNode5) {
  // Solve the left and right child nodes of the root, by branching on x4
  const auto root = ConstructMILPExample2RootNode();
  root->Branch(4);
  SolveNodeWithDefaultSettings(root->left_child());
  SolveNodeWithDefaultSettings(root->right_child());

  const scs_float x_expected_l[4] = {0.7, 1, 1, 1.4};
  TestNodeSolve(*(root->left_child()), SCS_SOLVED, -4.9, x_expected_l, false,
                2E-2);

  const scs_float x_expected_r[4] = {0.2, 2.0 / 3.0, 1, 1.4};
  TestNodeSolve(*(root->right_child()), SCS_SOLVED, -47.0 / 30, x_expected_r,
                false, 2E-2);
}

std::unique_ptr<ScsBranchAndBoundTest> ConstructScsBranchAndBoundTest(
    const MIPdata& data) {
  SCS_PROBLEM_DATA scs_data;
  scs_data.A = data.A_.get();
  scs_data.b = data.b_.get();
  scs_data.c = data.c_.get();
  scs_data.m = data.A_->m;
  scs_data.n = data.A_->n;
  scs_data.stgs = static_cast<SCS_SETTINGS*>(scs_malloc(sizeof(SCS_SETTINGS)));
  SetScsSettingToDefault(scs_data.stgs);

  auto test = std::make_unique<ScsBranchAndBoundTest>(
      scs_data, *(data.cone_), data.d_, data.binary_var_indices_);

  scs_free(scs_data.stgs);
  return test;
}

std::unique_ptr<ScsBranchAndBoundTest> ConstructScsBranchAndBoundMILP1Test() {
  const MIPdata data = ConstructMILPExample1();
  return ConstructScsBranchAndBoundTest(data);
}

std::unique_ptr<ScsBranchAndBoundTest> ConstructScsBranchAndBoundMILP2Test() {
  const MIPdata data = ConstructMILPExample2();
  return ConstructScsBranchAndBoundTest(data);
}

std::unique_ptr<ScsBranchAndBoundTest> ConstructScsBranchAndBoundMILP3Test() {
  const MIPdata data = ConstructMILPExample3();
  return ConstructScsBranchAndBoundTest(data);
}

GTEST_TEST(TestScsBranchAndBound, TestConstructor) {
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  EXPECT_EQ(dut->best_upper_bound(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(dut->best_lower_bound(), -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(dut->active_leaves().empty());
  const MIPdata data = ConstructMILPExample2();
  IsConeEqual(*(dut->cone()), *(data.cone_));
}

GTEST_TEST(TestScsBranchAndBound, TestPickBranchingVariable) {
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->root()->Solve(dut->scs_settings());
  // The optimal solution to the root is (0.7, 1, 1, 1.4, 0)
  // If we pick the most ambivalent branching variable, then we should return x0
  // If we pick the least ambivalent branching variable, then we should return
  // either x2 or x4.
  dut->bnb_tree()->ChoosePickBranchingVariableMethod(
      ScsBranchAndBound::PickVariable::MostAmbivalent);
  EXPECT_EQ(dut->PickBranchingVariable(*(dut->root())), 0);
  dut->bnb_tree()->ChoosePickBranchingVariableMethod(
      ScsBranchAndBound::PickVariable::LeastAmbivalent);
  EXPECT_TRUE(dut->PickBranchingVariable(*(dut->root())) == 2 ||
              dut->PickBranchingVariable(*(dut->root())) == 4);
  EXPECT_THROW(dut->bnb_tree()->ChoosePickBranchingVariableMethod(
                   ScsBranchAndBound::PickVariable::UserDefined),
               std::runtime_error);

  // Test user-defined method to pick the branching variable.
  dut->bnb_tree()->SetUserDefinedBranchingVariableMethod(
      [](const ScsNode& node) -> int {
        return *(node.binary_var_indices().begin());
      });
  EXPECT_EQ(dut->PickBranchingVariable(*(dut->root())), 0);

  dut->bnb_tree()->SetUserDefinedBranchingVariableMethod(
      [](const ScsNode& node) -> int {
        return *(++node.binary_var_indices().begin());
      });
  EXPECT_EQ(dut->PickBranchingVariable(*(dut->root())), 2);
}

template<typename T>
bool IsVectorEqual(const std::vector<T>& vec1, const std::vector<T>& vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  return std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

GTEST_TEST(TestScsBranchAndBound, TestSolveNode1) {
  auto dut = ConstructScsBranchAndBoundMILP1Test();
  EXPECT_EQ(dut->best_upper_bound(), std::numeric_limits<double>::infinity());
  dut->SolveRootNode();
  const scs_float x_expected[4] = {0, 1, 0, 0.5};
  TestNodeSolve(*(dut->root()), SCS_SOLVED, 1.5, x_expected, true, 1E-3);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()), x_expected), {0, 1, 0, 0.5});

  // The best lower bound should be the cost on the root node.
  EXPECT_NEAR(dut->best_lower_bound(), dut->root()->cost(), 1E-10);
  // Since the root node solution is integral, the best upper bound should also
  // be the root node cost.
  EXPECT_NEAR(dut->best_upper_bound(), dut->root()->cost(), 1E-10);
  // The root node is fathomed since the optimal solution is integral.
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root())));

  EXPECT_TRUE(dut->IsConverged());
  EXPECT_TRUE(dut->active_leaves().empty());
}

GTEST_TEST(TestScsBranchAndBound, TestSolveNode2) {
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();
  const scs_float x_expected[5] = {0.7, 1, 1, 1.4, 0};
  TestNodeSolve(*(dut->root()), SCS_SOLVED, -4.9, x_expected, false, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()), x_expected), {0.7, 1, 1, 1.4, 0});
  // The best lower bound should be the cost on the root node.
  EXPECT_NEAR(dut->best_lower_bound(), dut->root()->cost(), 1E-10);
  // The root node is not fathomed.
  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root())));
  EXPECT_TRUE(IsListEqualAfterReshuffle(dut->active_leaves(), {dut->root()}));
}

GTEST_TEST(TestScsBranchAndBound, TestSolveNode3) {
  auto dut = ConstructScsBranchAndBoundMILP3Test();
  dut->SolveRootNode();
  TestNodeSolve(*(dut->root()), SCS_UNBOUNDED,
                -std::numeric_limits<double>::infinity(), nullptr, false,
                1E-10);

  EXPECT_EQ(dut->best_lower_bound(), -std::numeric_limits<double>::infinity());
  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root())));
  EXPECT_TRUE(IsListEqualAfterReshuffle(dut->active_leaves(), {dut->root()}));
}

GTEST_TEST(TestScsBranchAndBound, TestBranchAndSolve1) {
  // Branch the root node at x0.
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();
  dut->BranchAndSolve(dut->root(), 0);

  // The left node is infeasible.
  TestNodeSolve(*(dut->root()->left_child()), SCS_INFEASIBLE, NAN, nullptr, NAN,
                1E-3);
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->left_child())));

  // The right node finds an integral solution. Since the left node is
  // infeasible, the right node solution is also the solution to the
  // mixed-integer problem.
  const scs_float x_expected_r[4] = {1.0 / 3.0, 1, 1, 0};
  TestNodeSolve(*(dut->root()->right_child()), SCS_SOLVED, -13.0 / 3.0,
                x_expected_r, true, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->right_child()), x_expected_r), {1, 1.0 / 3.0, 1, 1, 0});

  EXPECT_NEAR(dut->best_lower_bound(), dut->root()->right_child()->cost(),
              1E-10);
  EXPECT_NEAR(dut->best_upper_bound(), dut->root()->right_child()->cost(),
              1E-10);
  EXPECT_TRUE(dut->IsConverged());
  EXPECT_TRUE(dut->active_leaves().empty());
}

GTEST_TEST(TestScsBranchAndBound, TestBranchAndSolve2) {
  // Branch the root node at x2
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();
  dut->BranchAndSolve(dut->root(), 2);

  const scs_float x_expected_l[4] = {1, 2.0 / 3.0, 1, 1};
  TestNodeSolve(*(dut->root()->left_child()), SCS_SOLVED, 23.0 / 6.0,
                x_expected_l, true, 1E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->left_child()), x_expected_l), {1, 2.0 / 3.0, 0, 1, 1});

  const scs_float x_expected_r[4] = {0.7, 1, 1.4, 0};
  TestNodeSolve(*(dut->root()->right_child()), SCS_SOLVED, -4.9, x_expected_r,
                false, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->right_child()), x_expected_r), {0.7, 1, 1, 1.4, 0});


  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->left_child())));
  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root()->right_child())));
  // The left child finds an integral solution.
  EXPECT_NEAR(dut->best_upper_bound(), dut->root()->left_child()->cost(),
              1E-10);
  EXPECT_NEAR(dut->best_lower_bound(), dut->root()->right_child()->cost(),
              1E-10);
  EXPECT_FALSE(dut->IsConverged());
  EXPECT_TRUE(IsListEqualAfterReshuffle(
      dut->active_leaves(), {dut->root()->right_child()}));
}

GTEST_TEST(TestScsBranchAndBound, TestBranchAndSolve3) {
  // Branch the root node at x4
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();
  dut->BranchAndSolve(dut->root(), 4);

  const scs_float x_expected_l[4] = {0.7, 1, 1, 1.4};
  TestNodeSolve(*(dut->root()->left_child()), SCS_SOLVED, -4.9, x_expected_l,
                false, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->left_child()), x_expected_l), {0.7, 1, 1, 1.4, 0});


  const scs_float x_expected_r[4] = {0.2, 2.0 / 3.0, 1, 1.4};
  TestNodeSolve(*(dut->root()->right_child()), SCS_SOLVED, -47.0 / 30,
                x_expected_r, false, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->right_child()), x_expected_r), {0.2, 2.0 / 3.0, 1, 1.4, 1});


  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root()->left_child())));
  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root()->right_child())));

  EXPECT_EQ(dut->best_upper_bound(), std::numeric_limits<double>::infinity());
  EXPECT_NEAR(dut->best_lower_bound(), dut->root()->left_child()->cost(),
              1E-10);
  EXPECT_FALSE(dut->IsConverged());
  EXPECT_TRUE(IsListEqualAfterReshuffle(
      dut->active_leaves(),
      {dut->root()->left_child(), dut->root()->right_child()}));

  // Branch the left node on x0
  dut->BranchAndSolve(dut->root()->left_child(), 0);
  // root->left->left is infeasible.
  TestNodeSolve(*(dut->root()->left_child()->left_child()), SCS_INFEASIBLE, std::numeric_limits<double>::infinity(), nullptr, false, 1E-10);
  // root->left->right finds an integral solution (1, 1/3, 1, 1, 0), with cost
  // -13/3. This should update the best upper bound to -13/3, and thus the node
  // root->right should become fathomed (it has cost -47/30).
  const scs_float x_expected_l_r[3] = {1.0 / 3.0, 1, 1};
  TestNodeSolve(*(dut->root()->left_child()->right_child()), SCS_SOLVED, -13.0 / 3.0, x_expected_l_r, true, 1E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->left_child()->right_child()), x_expected_l_r), {1, 1.0 / 3.0, 1, 1, 0});
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->left_child()->left_child())));
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->left_child()->right_child())));
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->right_child())));

  EXPECT_EQ(dut->best_lower_bound(), dut->root()->left_child()->right_child()->cost());
  EXPECT_EQ(dut->best_upper_bound(), dut->root()->left_child()->right_child()->cost());
  EXPECT_TRUE(dut->IsConverged());
  EXPECT_TRUE(IsListEqualAfterReshuffle(dut->active_leaves(), {}));
}

GTEST_TEST(TestScsBranchAndBound, TestBranch4) {
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();

  dut->BranchAndSolve(dut->root(), 4);
  // root->r has cost -4.9
  // root->l has cost -1.5

  dut->BranchAndSolve(dut->root()->right_child(), 0);
  // root->r->l is infeasible.
  TestNodeSolve(*(dut->root()->right_child()->left_child()), SCS_INFEASIBLE, std::numeric_limits<double>::infinity(), nullptr, false, 0);
  // root->r->r has integral solution (1, -10/9, 1, 1/3, 1), with optimal cost -1/18
  const scs_float x_expected_r_r[3] = {-10.0 / 9.0, 1, 1.0 / 3.0};
  TestNodeSolve(*(dut->root()->right_child()->right_child()), SCS_SOLVED, -1.0 / 18.0, x_expected_r_r, true, 2E-2);
  IsVectorEqual(dut->RecoverSolutionFromNode(*(dut->root()->right_child()->right_child()), x_expected_r_r), {1, -10.0/9, 1, 1.0/3, 1});
  // There is only one active leaves, root->l with cost -4.9.
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->right_child()->right_child())));
  EXPECT_TRUE(dut->IsNodeFathomed(*(dut->root()->right_child()->left_child())));
  EXPECT_FALSE(dut->IsNodeFathomed(*(dut->root()->left_child())));
  EXPECT_TRUE(IsListEqualAfterReshuffle(dut->active_leaves(), {dut->root()->left_child()}));
  EXPECT_EQ(dut->best_upper_bound(), dut->root()->right_child()->right_child()->cost());
  EXPECT_EQ(dut->best_lower_bound(), dut->root()->left_child()->cost());
  EXPECT_FALSE(dut->IsConverged());
}

// Return the right-most active leaf.
ScsNode* RightMostActiveLeaf(const ScsBranchAndBound& bnb,
                             const ScsNode& root) {
  if (root.IsLeaf()) {
    return bnb.IsNodeFathomed(root) ? nullptr : const_cast<ScsNode*>(&root);
  }
  auto node = RightMostActiveLeaf(bnb, *(root.right_child()));
  return node ? node : RightMostActiveLeaf(bnb, *(root.left_child()));
}

GTEST_TEST(TestScsBranchAndBound, TestPickBranchingNode) {
  auto dut = ConstructScsBranchAndBoundMILP2Test();
  dut->SolveRootNode();

  // There is only one node, the root node.
  ScsNode* pick_node = dut->PickBranchingNode();
  EXPECT_EQ(pick_node, dut->root());

  dut->BranchAndSolve(dut->root(), 4);
  // Now there are two active nodes, the left node with optimal cost
  // -4.9, the right node with optimal cost -47/30.
  dut->bnb_tree()->ChoosePickBranchingNodeMethod(
      ScsBranchAndBound::PickNode::MinLowerBound);
  EXPECT_EQ(dut->PickBranchingNode(), dut->root()->left_child());
  // Pick the right most leaf node.
  std::function<ScsNode*(const ScsNode&)> rightmost_userfun =
      [&dut](const ScsNode& node) {
        return RightMostActiveLeaf(*(dut->bnb_tree()), node);
      };
  dut->bnb_tree()->SetUserDefinedBranchingNodeMethod(rightmost_userfun);
  EXPECT_EQ(dut->PickBranchingNode(), dut->root()->right_child());
}

GTEST_TEST(TestScsBranchAndBound, TestSolve1) {
  auto dut = ConstructScsBranchAndBoundMILP1Test();
  auto bnb_status = dut->bnb_tree()->Solve();
  EXPECT_EQ(bnb_status, SCS_SOLVED);
  EXPECT_TRUE(dut->IsConverged());
  EXPECT_NEAR(dut->best_upper_bound(), 1.5, 1E-2);
  EXPECT_NEAR(dut->best_lower_bound(), 1.5, 1E-2);
}

GTEST_TEST(TestScsBranchAndBound, TestSolve2) {
  for (auto pick_node : {ScsBranchAndBound::PickNode::DepthFirst, ScsBranchAndBound::PickNode::MinLowerBound}) {
    for (auto pick_variable : {ScsBranchAndBound::PickVariable::LeastAmbivalent, ScsBranchAndBound::PickVariable::MostAmbivalent}) {
      auto dut = ConstructScsBranchAndBoundMILP2Test();
      dut->bnb_tree()->ChoosePickBranchingNodeMethod(pick_node);
      dut->bnb_tree()->ChoosePickBranchingVariableMethod(pick_variable);
      auto bnb_status = dut->bnb_tree()->Solve();
      EXPECT_EQ(bnb_status, SCS_SOLVED);
      EXPECT_NEAR(dut->best_upper_bound(), -13.0 / 3, 1E-2);
      EXPECT_NEAR(dut->best_lower_bound(), -13.0 / 3, 1E-2);
      EXPECT_TRUE(dut->IsConverged());
    }
  }
}
}  // namespace
}  // namespace solvers
}  // namespace drake
