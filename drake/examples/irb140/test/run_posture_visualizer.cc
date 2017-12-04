#include "drake/examples/irb140/test/irb140_common.h"

int main() {
  auto robot = drake::examples::IRB140::ConstructIRB140();
  Eigen::Matrix<double, 6, 1> q;
  q.setZero();
  drake::examples::IRB140::VisualizePosture(*robot, q);
  return 0;
}