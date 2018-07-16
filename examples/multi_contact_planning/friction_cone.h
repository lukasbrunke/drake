#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace examples {
namespace multi_contact_planning {
class LinearizedFrictionCone {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(LinearizedFrictionCone)

  /**
   * Create a linearized friction about a unit length normal vector.
   */
  LinearizedFrictionCone(int num_edges,
                         const Eigen::Ref<const Eigen::Vector3d>& unit_normal,
                         double mu);

  int num_edges() const { return num_edges_; }

  const Eigen::Matrix3Xd& edges() const { return edges_; }

  const Eigen::Vector3d& unit_normal() const { return unit_normal_; }

 private:
  int num_edges_;
  Eigen::Vector3d unit_normal_;
  Eigen::Matrix3Xd edges_;
};
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
