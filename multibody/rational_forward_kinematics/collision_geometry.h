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
  kCapsule,
};

class CollisionGeometry {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CollisionGeometry)

  /**
   * @param X_BG The relative pose of the geometry on the frame.
   */
  CollisionGeometry(CollisionGeometryType type, const geometry::Shape* geometry,
                    multibody::BodyIndex body_index, geometry::GeometryId id,
                    math::RigidTransformd X_BG);

  CollisionGeometryType type() const { return type_; }

  const geometry::Shape& geometry() const { return *geometry_; }

  multibody::BodyIndex body_index() const { return body_index_; }

  geometry::GeometryId id() const { return id_; }

  const math::RigidTransformd& X_BG() const { return X_BG_; }

 private:
  CollisionGeometryType type_;
  const geometry::Shape* geometry_;
  multibody::BodyIndex body_index_;
  geometry::GeometryId id_;
  math::RigidTransformd X_BG_;
};

Eigen::Matrix3Xd GetVertices(const geometry::Shape& shape);
}  // namespace multibody
}  // namespace drake
