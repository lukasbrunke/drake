#pragma once

#include <memory>
#include <utility>

#include "drake/common/drake_copyable.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"

namespace drake {
namespace multibody {
enum class CollisionGeometryType {
  kPolytope,
  kSphere,
};

class CollisionGeometry {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CollisionGeometry)

  CollisionGeometry(CollisionGeometryType type,
                    std::unique_ptr<geometry::optimization::ConvexSet> geometry,
                    multibody::BodyIndex body_index, geometry::GeometryId id);

  CollisionGeometryType type() const { return type_; }

  const geometry::optimization::ConvexSet& geometry() const {
    return *geometry_;
  }

  multibody::BodyIndex body_index() const { return body_index_; }

  geometry::GeometryId id() const { return id_; }

 private:
  CollisionGeometryType type_;
  std::unique_ptr<geometry::optimization::ConvexSet> geometry_;
  multibody::BodyIndex body_index_;
  geometry::GeometryId id_;
};
}  // namespace multibody
}  // namespace drake
