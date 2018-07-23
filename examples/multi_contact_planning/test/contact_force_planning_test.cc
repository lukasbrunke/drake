#include "drake/examples/multi_contact_planning/contact_force_planning.h"

#include <gtest/gtest.h>

namespace drake {
namespace examples {
namespace multi_contact_planning {
class BoxTest : public ::testing::Test {
 public:
  BoxTest() {
    mass_ = 2;
    // clang-format off
    I_B_ << 0.02, 0, 0,
            0, 0.03, 0,
            0, 0, 0.01;
    // clang-format on
    const double mu = 1;
    const Eigen::Vector3d box_size(0.4, 0.5, 0.3);
    const int num_friction_edges = 6;
    for (int normal_axis = 0; normal_axis < 3; ++normal_axis) {
      for (double direction : {-1, 1}) {
        Eigen::Vector3d face_center = Eigen::Vector3d::Zero();
        face_center(normal_axis) = box_size(normal_axis) / 2 * direction;
        Eigen::Vector3d face_normal = Eigen::Vector3d::Zero();
        face_normal(normal_axis) = 1 * direction;
        candidate_contact_points_.emplace_back(face_center, num_friction_edges,
                                               face_normal, mu);
        // Add four points between the face center and each face corner.
        for (double scale1 : {-1.0 / 4, 1.0 / 4}) {
          for (double scale2 : {-1.0 / 4.0, 1.0 / 4.0}) {
            Eigen::Vector3d pt = box_size;
            pt(normal_axis) *= 0.5 * direction;
            pt((normal_axis + 1) % 3) *= scale1;
            pt((normal_axis + 1) % 3) *= scale2;
            candidate_contact_points_.emplace_back(pt, num_friction_edges,
                                                   face_normal, mu);
          }
        }
      }
    }
  }

 protected:
  double mass_;
  Eigen::Matrix3d I_B_;
  std::vector<ContactPoint> candidate_contact_points_;
};

TEST_F(BoxTest, TestConstructor) {
  const int nT = 5;
  const int num_arm_points = 3;
  double max_normal_contact_force = mass_ * 9.81;
  ContactForcePlanning prog(nT, mass_, I_B_, candidate_contact_points_,
                            num_arm_points, max_normal_contact_force);

}
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
