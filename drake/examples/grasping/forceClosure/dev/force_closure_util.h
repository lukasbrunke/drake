#include <Eigen/Core>

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
/**
 * Returns a polytopic inner approximation of the unit sphere in 6 dimensional
 * space. This polytope has 7 evenly spaced vertices.
 * The computation is adapted from section II.E of
 * Fast Computation of Optimal Contact Forces by Stephen Boyd and Ben Wegbreit
 */
Eigen::Matrix<double, 6, 7> GenerateWrenchPolytopeInnerSphere7Vertices();

/**
 * Returns a polytopic inner approximation of the unit sphere in 6 dimensional
 * space. This polytope has 12 evenly spaced vertices.
 * @return W    W(j, 2*i) = 0 for j ≠ i
 *              W(i, 2*i) = 1
 *              W(j, 2*i+1) = 0 for j ≠ i
 *              W(i, 2*i+1) = -1
 */
Eigen::Matrix<double, 6, 12> GenerateWrenchPolytopeInnerSphere12Vertices();
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake
