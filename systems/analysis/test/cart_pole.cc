#include "drake/systems/analysis/test/cart_pole.h"

#include <iomanip>
#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

void TrigPolyDynamics(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* G, symbolic::Polynomial* d) {
  const Eigen::Matrix<symbolic::Expression, 5, 1> x_expr =
      x.cast<symbolic::Expression>();
  const Matrix2<symbolic::Expression> M_expr =
      CartpoleMassMatrix<symbolic::Expression>(params, x_expr);
  *d = symbolic::Polynomial(M_expr(0, 0) * M_expr(1, 1) -
                            M_expr(1, 0) * M_expr(0, 1));
  const symbolic::Polynomial s(x(1));
  const symbolic::Polynomial c(x(2) - 1);
  (*f)(0) = symbolic::Polynomial(x(3)) * (*d);
  (*f)(1) = c * symbolic::Polynomial(x(4)) * (*d);
  (*f)(2) = -s * symbolic::Polynomial(x(4)) * (*d);

  Matrix2<symbolic::Expression> M_adj_expr;
  M_adj_expr << M_expr(1, 1), -M_expr(1, 0), -M_expr(0, 1), M_expr(0, 0);
  const Vector2<symbolic::Expression> f_tail_expr =
      M_adj_expr * (CalcCartpoleGravityVector<symbolic::Expression>(params, x_expr) -
                    CalcCartpoleBiasTerm<symbolic::Expression>(params, x_expr));
  (*f)(3) = symbolic::Polynomial(f_tail_expr(0));
  (*f)(4) = symbolic::Polynomial(f_tail_expr(1));
  (*G)(0) = symbolic::Polynomial();
  (*G)(1) = symbolic::Polynomial();
  (*G)(2) = symbolic::Polynomial();
  (*G)(3) = symbolic::Polynomial(M_adj_expr(0, 0));
  (*G)(4) = symbolic::Polynomial(M_adj_expr(1, 0));
}

symbolic::Polynomial CartpoleStateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x) {
  return symbolic::Polynomial(x(1) * x(1) + x(2) * x(2) - 2 * x(2));
}

controllers::LinearQuadraticRegulatorResult SynthesizeCartpoleTrigLqr(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<double, 5, 5>>& Q, double R) {
  const Eigen::Matrix<double, 6, 1> xu_des = Vector6d::Zero();
  const auto xu_des_ad = math::InitializeAutoDiff(xu_des);
  Eigen::Matrix<AutoDiffXd, 5, 1> n;
  AutoDiffXd d;
  CartpoleTrigDynamics<AutoDiffXd>(params, xu_des_ad.head<5>(), xu_des_ad(5),
                                   &n, &d);
  const Eigen::Matrix<AutoDiffXd, 5, 1> xdot_ad = n / d;
  const auto xdot_grad = math::ExtractGradient(xdot_ad);
  Eigen::Matrix<double, 1, 5> F;
  F << 0, 0, 1, 0, 0;
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      xdot_grad.leftCols<5>(), xdot_grad.col(5), Q, Vector1d(R),
      Eigen::MatrixXd(0, 1), F);
  return lqr_result;
}

template <typename T>
CartpoleTrigStateConverter<T>::CartpoleTrigStateConverter()
    : LeafSystem<T>(SystemTypeTag<CartpoleTrigStateConverter>{}) {
  this->DeclareInputPort("state", systems::PortDataType::kVectorValued, 4);
  this->DeclareVectorOutputPort("x_trig", 5,
                                &CartpoleTrigStateConverter<T>::CalcTrigState);
}

template <typename T>
void CartpoleTrigStateConverter<T>::CalcTrigState(const Context<T>& context,
                                            BasicVector<T>* x_trig) const {
  const Vector4<T> x_orig = this->get_input_port().Eval(context);
  x_trig->get_mutable_value() = ToCartpoleTrigState<T>(x_orig);
}

CartpoleClfController::CartpoleClfController(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Polynomial, 5, 1>>& f,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Polynomial, 5, 1>>& G,
    const symbolic::Polynomial& dynamics_denominator, symbolic::Polynomial V,
    double kappa, double u_max)
    : ClfController(x, f, G, dynamics_denominator, V, kappa), u_max_{u_max} {}

void CartpoleClfController::DoCalcControl(const Context<double>& context,
                                          BasicVector<double>* output) const {
  const Eigen::VectorXd x_val =
      this->get_input_port(this->x_input_index()).Eval(context);
  symbolic::Environment env;
  env.insert(x(), x_val);

  solvers::MathematicalProgram prog;
  auto u = prog.NewContinuousVariables(1, "u");
  prog.AddBoundingBoxConstraint(-u_max_, u_max_, u(0));
  prog.AddQuadraticCost(Eigen::Matrix<double, 1, 1>::Ones(), Vector1d(0), 0, u);

  double dVdx_times_f_val;
  Eigen::RowVectorXd dVdx_times_G_val;
  double dynamics_denominator_val;
  CalcVdot(env, &dVdx_times_f_val, &dVdx_times_G_val,
           &dynamics_denominator_val);
  const double V_val = V().Evaluate(env);
  // dVdx * G * u + dVdx * f <= -eps * V * n(x)
  AddClfConstraint(&prog, dVdx_times_f_val, dVdx_times_G_val,
                   dynamics_denominator_val, V_val, u);
  const double vdot_cost = 10;
  prog.AddLinearCost(dVdx_times_G_val / dynamics_denominator_val * vdot_cost,
                     dVdx_times_f_val / dynamics_denominator_val * vdot_cost,
                     u);
  const auto result = solvers::Solve(prog);
  if (!result.is_success()) {
    drake::log()->info(
        "dVdx*f+eps*V={}, dVdx*G={}",
        dVdx_times_f_val / dynamics_denominator_val + kappa() * V_val,
        dVdx_times_G_val / dynamics_denominator_val);
    drake::log()->error("ClfController fails at t={} with x={}, V={}",
                        context.get_time(), x_val.transpose(), V_val);
    DRAKE_DEMAND(result.is_success());
  }
  const Eigen::VectorXd u_val = result.GetSolution(u);
  output->get_mutable_value() = u_val;
}

void Simulate(const CartPoleParams& parameters,
              const Eigen::Matrix<symbolic::Variable, 5, 1>& x,
              const symbolic::Polynomial& clf, double u_bound, double kappa,
              const Eigen::Vector4d& initial_state, double duration) {
  systems::DiagramBuilder<double> builder;

  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  auto cart_pole = builder.AddSystem<multibody::MultibodyPlant<double>>(0.);
  multibody::Parser(cart_pole, scene_graph)
      .AddModelFromFile(FindResourceOrThrow(
          "drake/examples/multibody/cart_pole/cart_pole.sdf"));
  cart_pole->Finalize();
  builder.Connect(
      cart_pole->get_geometry_poses_output_port(),
      scene_graph->get_source_pose_port(cart_pole->get_source_id().value()));
  auto meshcat = std::make_shared<geometry::Meshcat>();
  geometry::MeshcatVisualizerParams meshcat_params{};
  meshcat_params.role = geometry::Role::kIllustration;
  auto visualizer = &geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, *scene_graph, meshcat, meshcat_params);
  unused(visualizer);

  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(parameters, x, &f, &G, &dynamics_denominator);

  auto clf_controller = builder.AddSystem<CartpoleClfController>(
      x, f, G, dynamics_denominator, clf, kappa, u_bound);
  auto state_logger =
      LogVectorOutput(cart_pole->get_state_output_port(), &builder);
  auto clf_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->clf_output_index()),
      &builder);
  auto control_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      &builder);
  unused(control_logger);
  auto trig_state_converter = builder.AddSystem<CartpoleTrigStateConverter<double>>();
  builder.Connect(cart_pole->get_state_output_port(),
                  trig_state_converter->get_input_port());
  builder.Connect(
      trig_state_converter->get_output_port(),
      clf_controller->get_input_port(clf_controller->x_input_index()));
  builder.Connect(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      cart_pole->get_actuation_input_port());

  symbolic::Environment env;
  env.insert(x, ToCartpoleTrigState<double>(initial_state));
  std::cout << std::setprecision(10)
            << "V(initial_state): " << clf.Evaluate(env) << "\n";
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  Simulator<double> simulator(*diagram);
  // ResetIntegratorFromFlags(&simulator, "implicit_euler", 0.0002);
  simulator.get_mutable_context().SetContinuousState(initial_state);
  diagram->Publish(simulator.get_context());
  std::cout << "Refresh meshcat brower and press to continue\n";
  std::cin.get();

  simulator.AdvanceTo(duration);
  std::cout << "finish simulation\n";

  std::cout << fmt::format(
      "final state: {}, final V: {}\n",
      state_logger->FindLog(simulator.get_context())
          .data()
          .rightCols<1>()
          .transpose(),
      clf_logger->FindLog(simulator.get_context()).data().rightCols<1>());
  // std::cout << "V: " << std::setprecision(10)
  //          << clf_logger->FindLog(simulator.get_context()).data() << "\n";
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::analysis::CartpoleTrigStateConverter)
