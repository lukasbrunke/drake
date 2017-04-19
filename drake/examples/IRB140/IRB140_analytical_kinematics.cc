#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

#include <queue>

#include "drake/common/drake_path.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree_construction.h"

using Eigen::Isometry3d;
using Eigen::Matrix;
using drake::symbolic::Variable;
using drake::symbolic::Expression;

namespace drake {
namespace examples {
namespace IRB140 {
IRB140AnalyticalKinematics::IRB140AnalyticalKinematics()
    : robot_(std::make_unique<RigidBodyTreed>()),
      l0_(0.1095),
      l1_x_(0.07),
      l1_y_(0.2425),
      l2_(0.36),
      l3_(0.2185),
      l4_(0.1615),
      c_{}, s_{},
      l0_var_("l0"),
      l1_x_var_("l1x"),
      l1_y_var_("l1y"),
      l2_var_("l2"),
      l3_var_("l3"),
      l4_var_("l4"),
      c23_var_("c23"),
      s23_var_("s23") {
  const std::string model_path = drake::GetDrakePath() + "/examples/IRB140/urdf/irb_140_shift.urdf";
  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      nullptr,
      robot_.get());

  for (int i = 0; i < 6; ++i) {
    c_[i] = symbolic::Variable("c" + std::to_string(i + 1));
    s_[i] = symbolic::Variable("s" + std::to_string(i + 1));
  }
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_01() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[0], 0, -s_[0], 0,
       s_[0], 0, c_[0], 0,
       0, -1, 0, l0_var_,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_12() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[1], -s_[1], 0, l1_x_var_,
       s_[1], c_[1], 0, -l1_y_var_,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_23() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[2], -s_[2], 0, 0,
       s_[2], c_[2], 0, -l2_var_,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_13() const {
  Eigen::Matrix<symbolic::Expression, 4, 4> X;
  X << c23_var_, -s23_var_, 0, l1_x_var_ + s_[1] * l2_var_,
       s23_var_, c23_var_, 0, -l1_y_var_ - c_[1] * l2_var_,
      0, 0, 1, 0,
      0, 0, 0, 1;
  return X;
};

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_34() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << 1, 0, 0, l3_var_,
       0, c_[3], -s_[3], 0,
       0, s_[3], c_[3], 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_45() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[4], s_[4], 0, l4_var_,
       -s_[4], c_[4], 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
};

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_56() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << 1, 0, 0, 0,
       0, c_[5], -s_[5], 0,
       0, s_[5], c_[5], 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
};

Eigen::Isometry3d EvalIsometry3dFromExpression(const Eigen::Matrix<symbolic::Expression, 4, 4>& X_sym, const symbolic::Environment& env) {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R(i, j) = X_sym(i, j).Evaluate(env);
    }
    t(i) = X_sym(i, 3).Evaluate(env);
  }
  Eigen::Isometry3d X_val;
  X_val.translation() = t;
  X_val.linear() = R;
  return X_val;
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_01(double theta) const {
  symbolic::Environment env;
  env.insert(l0_var_, l0_);
  env.insert(c_[0], std::cos(theta));
  env.insert(s_[0], std::sin(theta));
  const auto& X_sym = X_01();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_12(double theta) const {
  symbolic::Environment env;
  env.insert(l1_x_var_, l1_x_);
  env.insert(l1_y_var_, l1_y_);
  env.insert(c_[1], std::cos(theta));
  env.insert(s_[1], std::sin(theta));
  const auto& X_sym = X_12();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_23(double theta) const {
  symbolic::Environment env;
  env.insert(l2_var_, l2_);
  env.insert(c_[2], std::cos(theta));
  env.insert(s_[2], std::sin(theta));
  const auto& X_sym = X_23();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_13(double theta2, double theta3) const {
  symbolic::Environment env;
  env.insert(c23_var_, std::cos(theta2 + theta3));
  env.insert(s23_var_, std::sin(theta2 + theta3));
  env.insert(c_[1], std::cos(theta2));
  env.insert(s_[1], std::sin(theta2));
  env.insert(l1_x_var_, l1_x_);
  env.insert(l1_y_var_, l1_y_);
  env.insert(l2_var_, l2_);
  const auto& X_sym = X_13();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_34(double theta) const {
  symbolic::Environment env;
  env.insert(l3_var_, l3_);
  env.insert(c_[3], std::cos(theta));
  env.insert(s_[3], std::sin(theta));
  const auto& X_sym = X_34();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_45(double theta) const {
  symbolic::Environment env;
  env.insert(l4_var_, l4_);
  env.insert(c_[4], std::cos(theta));
  env.insert(s_[4], std::sin(theta));
  const auto& X_sym = X_45();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_56(double theta) const {
  symbolic::Environment env;
  env.insert(c_[5], std::cos(theta));
  env.insert(s_[5], std::sin(theta));
  const auto& X_sym = X_56();
  return EvalIsometry3dFromExpression(X_sym, env);
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_06(const Eigen::Matrix<double, 6, 1>& q) const {
  return X_01(q(0)) * X_12(q(1)) * X_23(q(2)) * X_34(q(3)) * X_45(q(4)) * X_56(q(5));
}

// If the abs value of x is small than 1 + tol, then set x to std::max(std::min(1, x), -1)
// else, leave x what it is
double clapToPlusMinusOneRange(double x, double tol = 1E-6) {
  if (std::abs(x) < 1 + 1E-6) {
    return std::max(std::min(x, 1.0), -1.0);
  } else {
    return x;
  }
}

std::vector<double> IRB140AnalyticalKinematics::q1(const Eigen::Isometry3d& link6_pose) const {
  double theta = std::atan2(link6_pose.translation()(1), link6_pose.translation()(0));
  std::vector<double> q1_all;
  for (int i = -2; i <= 2; ++i) {
    double q1_val = theta + M_PI * i;
    if ( q1_val >= robot_->joint_limit_min(0) && q1_val <= robot_->joint_limit_max(0)) {
      q1_all.push_back(q1_val);
    }
  }
  return q1_all;
}

std::vector<double> IRB140AnalyticalKinematics::q2(const Eigen::Isometry3d& link6_pose, double q1) const {
  std::vector<double> q2_all;
  double a0 = link6_pose.translation()(2) - l0_ - l1_y_;
  double b0;
  if (std::abs(std::cos(q1)) > 0.1) {
    b0 = link6_pose.translation()(0) / std::cos(q1) - l1_x_;
  } else {
    b0 = link6_pose.translation()(1) / std::sin(q1) - l1_x_;
  }
  // a0 * cos(q2) + b0 * sin(q2) = c0;
  double c0 = (a0 * a0 + b0 * b0 + l2_ * l2_ - std::pow(l3_ + l4_, 2.0)) / (2 * l2_);
  double sin_q2_plus_phi = c0 / std::sqrt(a0 * a0 + b0 * b0);
  if (std::abs(sin_q2_plus_phi) > 1) {
    return q2_all;
  }
  double phi = std::atan2(a0, b0);
  double q2_plus_phi = std::asin(sin_q2_plus_phi);
  for (int i = -1; i <= 1; ++i) {
    double q2_val = q2_plus_phi - phi + 2 * M_PI * i;
    if (q2_val >= robot_->joint_limit_min(1) && q2_val <= robot_->joint_limit_max(1)) {
      q2_all.push_back(q2_val);
    }
    q2_val = M_PI - q2_plus_phi - phi + 2 * M_PI * i;
    if (q2_val >= robot_->joint_limit_min(1) && q2_val <= robot_->joint_limit_max(1)) {
      q2_all.push_back(q2_val);
    }
  }
  return q2_all;
}

std::vector<double> IRB140AnalyticalKinematics::q3(const Eigen::Isometry3d& link6_pose,
                                                   double q1,
                                                   double q2) const {
  std::vector<double> q3_all;
  double a0 = link6_pose.translation()(2) - l0_ - l1_y_;
  double b0;
  if (std::abs(std::cos(q1)) > 0.1) {
    b0 = link6_pose.translation()(0) / std::cos(q1) - l1_x_;
  } else {
    b0 = link6_pose.translation()(1) / std::sin(q1) - l1_x_;
  }
  double cos_q23 = (b0 - l2_ * std::sin(q2)) / (l3_ + l4_);
  double sin_q23 = (a0 - l2_ * std::cos(q2)) / -(l3_ + l4_);
  double q2_plus_q3 = atan2(sin_q23, cos_q23);
  for (int i = -1; i <= 1; ++i) {
    double q3_val = q2_plus_q3 - q2 + 2 * M_PI * i;
    if (q3_val >= robot_->joint_limit_min(2) && q3_val <= robot_->joint_limit_max(2)) {
      q3_all.push_back(q3_val);
    }
  }
  return q3_all;
}

void add_q456_fun(double q4_val, double q6_val, double q5, RigidBodyTreed* robot, std::vector<Eigen::Vector3d>* q456_all) {
  if (q4_val >= robot->joint_limit_min(3) && q4_val <= robot->joint_limit_max(3)) {
    if (q6_val >= robot->joint_limit_min(5) && q6_val <= robot->joint_limit_max(5)) {
      for (int i = -1; i <= 1; ++i) {
        double q5_val = q5 + 2 * M_PI * i;
        if (q5_val >= robot->joint_limit_min(4) && q5_val <= robot->joint_limit_max(4)) {
          q456_all->emplace_back(q4_val, q5_val, q6_val);
        }
      }
    }
  }

}
std::vector<Eigen::Vector3d> IRB140AnalyticalKinematics::q456(const Eigen::Isometry3d& link6_pose, double q1, double q2, double q3) const {
  std::vector<Eigen::Vector3d> q456_all;
  double R11 = link6_pose.linear()(0, 0);
  double R21 = link6_pose.linear()(1, 0);
  double R31 = link6_pose.linear()(2, 0);
  double R12 = link6_pose.linear()(0, 1);
  double R22 = link6_pose.linear()(1, 1);
  double R32 = link6_pose.linear()(2, 1);
  double R13 = link6_pose.linear()(0, 2);
  double R23 = link6_pose.linear()(1, 2);
  double R33 = link6_pose.linear()(2, 2);
  double c1 = std::cos(q1);
  double c23 = std::cos(q2 + q3);
  double s1 = std::sin(q1);
  double s23 = std::sin(q2 + q3);
  double c5 = c1 * c23 * R11 + s1 * c23 * R21 - s23 * R31;
  if (std::abs(c5) > 1) {
    return q456_all;
  }
  double c4_times_s5;
  double s4_times_s5 = s1 * R11 - c1 * R21;
  if (std::abs(c23) > 1E-3) {
    c4_times_s5 = (R31 + s23 * c5) / c23;
  } else {
    c4_times_s5 = (c1 * R11 + s1 * R21 - c23 * c5) / s23;
  }
  if (std::abs(std::abs(c5) - 1) > 1E-6) {
    // s5 is not 0
    std::array<double, 2> s5 = {{std::sqrt(1 - c5 * c5), -std::sqrt(1 - c5 * c5)}};
    for (int i = 0; i < 2; ++i) {
      double c4 = c4_times_s5 / s5[i];
      double s4 = s4_times_s5 / s5[i];
      c4 = clapToPlusMinusOneRange(c4);
      s4 = clapToPlusMinusOneRange(s4);
      if (std::abs(c4) > 1 || std::abs(s4) > 1) {
        continue;
      }
      double theta4 = std::atan2(s4, c4);
      double theta5 = std::atan2(s5[i], c5);
      // Now solve theta6
      double s6, c6;
      // A6 * [s6;c6] = b6;
      Eigen::Matrix2d A6;
      Eigen::Vector2d b6;
      if (std::abs(c23) > 1E-3) {
        // Use R32 and R33 to compute theta6
        A6 << c23 * s4, -c23 * c4 * c5 - s23 * s5[i],
             c23 * c4 * c5 + s23 * s5[i], c23 * s4;
        b6 << R32, R33;
      } else {
        // Use R12 and R13 to compute theta6
        A6 << c1 * s23 * s4 - s1 * c4, c1 * (c23 * s5[i] - s23 * c4 * c5) - s1 * s4 * c5,
             -c1 * (c23 * s5[i] - s23 * c4 * c5) + s1 * s4 * c5, c1 * s23 * s4 - s1 * c4;
        b6 << R12, R13;
        if (std::abs(A6.determinant()) < 1E-3) {
          // Use R22 and R23 to compute theta6
          A6 << c1 * c4 + s1 * s23 * s4, c1 * s4 * c5 + s1 * (c23 * s5[i] - s23 * c4 * c5),
                -c1 * s4 * c5 - s1 * (c23 * s5[i] - s23 * c4 * c5), c1 * c4 + s1 * s23 * s4;
          b6 << R22, R23;
        }
      }
      Eigen::Vector2d sin_cos_q6 = A6.colPivHouseholderQr().solve(b6);
      s6 = sin_cos_q6(0);
      c6 = sin_cos_q6(1);
      s6 = clapToPlusMinusOneRange(s6);
      c6 = clapToPlusMinusOneRange(c6);
      if (std::abs(s6) > 1 || std::abs(c6) > 1) {
        continue;
      }
      double theta6 = std::atan2(s6, c6);
      for (int i4 = -1; i4 <= 1; ++i4) {
        double q4_val = theta4 + 2 * M_PI * i4;
        if (q4_val < robot_->joint_limit_min(3) || q4_val > robot_->joint_limit_max(3)) {
          continue;
        }
        for (int i5 = -1; i5 <= 1; ++i5) {
          double q5_val = theta5 + 2 * M_PI * i5;
          if (q5_val < robot_->joint_limit_min(4) || q5_val > robot_->joint_limit_max(4)) {
            continue;
          }
          for (int i6 = -1; i6 <= 1; ++i6) {
            double q6_val = theta6 + 2 * M_PI * i6;
            if (q6_val < robot_->joint_limit_min(5) || q6_val > robot_->joint_limit_max(5)) {
              continue;
            }
            q456_all.push_back(Eigen::Vector3d(q4_val, q5_val, q6_val));
          }
        }
      }
    }
  } else {
    // s5 = 0. Degenerate case.
    if (c5 > 0) {
      // c5 = 1, s5 = 0, we can only compute q4 + q6.
      // A * [sin_q4_plus_q6; cos_q4_plus_q6] = b
      Eigen::Matrix<double, 6, 2> A;
      Eigen::Matrix<double, 6, 1> b;
      // clang-format off
      A << -s1, -c1 * s23,
          c1, -s1 * s23,
          0, -c23,
          c1 * s23, -s1,
          s1 * s23, c1,
          c23, 0;
      // clang-format off
      b << R12, R22, R32, R13, R23, R33;
      Eigen::Vector2d sin_cos_q4_plus_q6 = A.colPivHouseholderQr().solve(b);
      double sin_q4_plus_q6 = sin_cos_q4_plus_q6(0);
      double cos_q4_plus_q6 = sin_cos_q4_plus_q6(1);
      double q4_plus_q6 = std::atan2(sin_q4_plus_q6, cos_q4_plus_q6);
      for (int i = -1; i <= 1; ++i) {
        double q4_plus_q6_val = q4_plus_q6 + 2 * M_PI * i;
        double q4_val = robot_->joint_limit_min(3);
        add_q456_fun(q4_val, q4_plus_q6_val - q4_val, 0, robot_.get(), &q456_all);
        q4_val = robot_->joint_limit_max(3);
        add_q456_fun(q4_val, q4_plus_q6_val - q4_val, 0, robot_.get(), &q456_all);
      }
    } else {
      // c5 = -1, s5 = 0, we can only compute q4 - q6;
      // A * [sin(q4-q6); cos(q4-q6)] = b
      Eigen::Matrix<double, 6, 2> A;
      Eigen::Matrix<double, 6, 1> b;
      // clang-format off
      A << s1, c1 * s23,
           -c1, s1 * s23,
           0, c23,
           c1 * c23, -s1,
           s1 * s23, c1,
           c23, 0;
      // clang-format off
      b << R12, R22, R32, R13, R23, R33;
      Eigen::Vector2d sin_cos_q4_minus_16 = A.colPivHouseholderQr().solve(b);
      double sin_q4_minus_q6 = sin_cos_q4_minus_16(0);
      double cos_q4_minus_q6 = sin_cos_q4_minus_16(1);
      double q4_minus_q6 = std::atan2(sin_q4_minus_q6, cos_q4_minus_q6);
      for (int i = -1; i <= 1; ++i) {
        double q4_minus_q6_val = q4_minus_q6 + 2 * M_PI * i;
        double q4_val = robot_->joint_limit_min(3);
        add_q456_fun(q4_val, -q4_minus_q6_val + q4_val, M_PI, robot_.get(), &q456_all);
        q4_val = robot_->joint_limit_max(3);
        add_q456_fun(q4_val, -q4_minus_q6_val + q4_val, M_PI, robot_.get(), &q456_all);
      }

    }
  }
  return q456_all;
}

std::vector<Eigen::Matrix<double, 6, 1>> IRB140AnalyticalKinematics::inverse_kinematics(const Eigen::Isometry3d& link6_pose) const {
  std::queue<Eigen::Matrix<double, 6, 1>> q_all;
  const auto& q1_all = q1(link6_pose);
  for (const auto& q1_val : q1_all) {
    Eigen::Matrix<double, 6, 1> q;
    q(0) = q1_val;
    q_all.push(q);
  }

  int q_all_size = q_all.size();
  for (int i = 0; i < q_all_size; ++i) {
    auto q = q_all.front();
    q_all.pop();
    const auto& q2_all = q2(link6_pose, q(0));
    for (const double q2_val : q2_all) {
      q(1) = q2_val;
      q_all.push(q);
    }
  }

  q_all_size = q_all.size();
  for (int i = 0; i < q_all_size; ++i) {
    auto q = q_all.front();
    q_all.pop();
    const auto& q3_all = q3(link6_pose, q(0), q(1));
    for (const double q3_val : q3_all) {
      q(2) = q3_val;
      q_all.push(q);
    }
  }

  q_all_size = q_all.size();
  for (int i = 0; i < q_all_size; ++i) {
    auto q = q_all.front();
    q_all.pop();
    const auto& q456_all = q456(link6_pose, q(0), q(1), q(2));
    for (const auto& q456_val : q456_all) {
      q(3) = q456_val(0);
      q(4) = q456_val(1);
      q(5) = q456_val(2);
      q_all.push(q);
    }
  }
/*
  int q_all_size = q_all.size();
  for (int i = 0; i < q_all_size; ++i) {
    Eigen::Matrix<double, 6, 1> q = q_all.front();
    q_all.pop();
    const auto& q23_all = q23(link6_pose, q(0));
    for (int j = 0; j < static_cast<int>(q23_all.size()); ++j) {
      q(1) = q23_all[j].first;
      q(2) = q23_all[j].second;
      q_all.push(q);
    }
  }
*/
  std::vector<Eigen::Matrix<double, 6, 1>> q_all_vec;
  q_all_vec.reserve(q_all.size());
  while (!q_all.empty()) {
    const Eigen::Matrix<double, 6, 1> q = q_all.front();
    q_all_vec.push_back(q);
    q_all.pop();
  }
  return q_all_vec;
};

}
}
}
