#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
// Store the information on the contact facet, such as the vertices
// the normal directions, the friction coefficient, etc.
class ContactFacet {
 public:
  ContactFacet(const Eigen::Matrix3Xd& vertices, const Eigen::Vector3d& face_normal);

  const Eigen::Matrix3Xd& vertices() const {return vertices_;};

  const Eigen::Vector3d& facet_normal() const {return facet_normal_;};

  int num_vertices() const {return vertices_.cols();}

  template<int kNumEdges>
  typename std::enable_if<kNumEdges != Eigen::Dynamic, Eigen::Matrix<double, 3, kNumEdges>>::type
  LinearizedFrictionConeEdges(double friction_coefficient) const {
    Eigen::Matrix<double, 3, kNumEdges> edges;
    Eigen::Matrix<double, 1, kNumEdges> theta = Eigen::Matrix<double, 1, kNumEdges + 1>::LinSpaced(kNumEdges + 1, 0, 2 *M_PI).head(kNumEdges);
    Eigen::Vector3d cone_axis(0, 0, 1);
    // Now rotate the friction cone, so that the cone axis aligns with the facet normal.
    if ((facet_normal_ - cone_axis).norm() < 1E4 * std::numeric_limits<double>::epsilon()) {
      cone_axis << 0, 1, 0;
      edges.row(0) = theta.array().sin().matrix() * friction_coefficient;
      edges.row(2) = theta.array().cos().matrix() * friction_coefficient;
      edges.row(1) = Eigen::Matrix<double, 1, kNumEdges>::Ones();

    } else {
      edges.row(0) = theta.array().sin().matrix() * friction_coefficient;
      edges.row(1) = theta.array().cos().matrix() * friction_coefficient;
      edges.row(2) = Eigen::Matrix<double, 1, kNumEdges>::Ones();
    }
    Eigen::Vector3d rotate_axis = cone_axis.cross(facet_normal_);
    double rotate_angle = std::acos(cone_axis.dot(facet_normal_));
    rotate_axis /= rotate_axis.norm();
    Eigen::AngleAxisd rotate(rotate_angle, rotate_axis);
    return rotate.toRotationMatrix() * edges;
  };

 private:
  Eigen::Matrix3Xd vertices_;
  Eigen::Vector3d facet_normal_;
};
}  // namespace forceClosure
}  // namespace examples
}  // namespace grasping
}  // namespace drake