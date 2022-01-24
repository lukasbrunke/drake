#include "drake/multibody/rational_forward_kinematics/collision_geometry.h"

namespace drake {
namespace multibody {
CollisionGeometry::CollisionGeometry(
    CollisionGeometryType type,
    std::unique_ptr<geometry::optimization::ConvexSet> geometry,
    multibody::BodyIndex body_index, geometry::GeometryId id)
    : type_{type},
      geometry_{std::move(geometry)},
      body_index_{body_index},
      id_{id} {}
}  // namespace multibody
}  // namespace drake
