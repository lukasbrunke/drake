#include <Eigen/Core>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace examples {
namespace multi_contact_planning {
class LinearizedFrictionCone {
 public:

 private:
   int num_edges_;
   Eigen::Matrix3Xd edges_;
};
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
