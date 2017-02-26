#include "drake/examples/grasping/forceClosure/dev/contact_facet.h"

#include "drake/common/drake_assert.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
ContactFacet::ContactFacet(const Eigen::Matrix3Xd &vertices,
                           const Eigen::Vector3d &face_normal)
    : vertices_(vertices), facet_normal_(face_normal / face_normal.norm()) {
  // Check if vertices are co-planar.
  DRAKE_DEMAND(vertices.size() >= 3);
  auto offset = face_normal.transpose() * vertices;
  for (int i = 1; i < offset.size(); ++i) {
    DRAKE_DEMAND(std::abs(offset(i) - offset(0)) < 1E-15);
  }
}
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake