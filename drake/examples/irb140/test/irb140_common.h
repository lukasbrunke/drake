#include <Eigen/Core>
#include <Eigen/Geometry>

namespace drake {
namespace examples {
namespace IRB140 {
bool CompareIsometry3d(const Eigen::Isometry3d& X1, const Eigen::Isometry3d& X2, double tol = 1E-10);
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
