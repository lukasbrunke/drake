#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/deprecation_pybind.h"
#include "drake/bindings/pydrake/common/serialize_pybind.h"
#include "drake/bindings/pydrake/common/wrap_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/integrator_base.h"
#include "drake/systems/analysis/monte_carlo.h"
#include "drake/systems/analysis/region_of_attraction.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_print_stats.h"

using std::unique_ptr;

namespace drake {
namespace pydrake {

PYBIND11_MODULE(analysis, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;

  m.doc() = "Bindings for the analysis portion of the Systems framework.";

  py::module::import("pydrake.systems.framework");

  {
    using Class = SimulatorConfig;
    constexpr auto& cls_doc = pydrake_doc.drake.systems.SimulatorConfig;
    py::class_<Class> cls(m, "SimulatorConfig", cls_doc.doc);
    cls  // BR
        .def(ParamInit<Class>());
    DefAttributesUsingSerialize(&cls, cls_doc);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }

  {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using Class = SimulatorStatus;
    constexpr auto& cls_doc = doc.SimulatorStatus;
    py::class_<Class> cls(m, "SimulatorStatus", cls_doc.doc);

    using Enum = Class::ReturnReason;
    constexpr auto& enum_doc = cls_doc.ReturnReason;
    py::enum_<Class::ReturnReason>(cls, "ReturnReason", enum_doc.doc)
        .value("kReachedBoundaryTime", Enum::kReachedBoundaryTime,
            enum_doc.kReachedBoundaryTime.doc)
        .value("kReachedTerminationCondition",
            Enum::kReachedTerminationCondition,
            enum_doc.kReachedTerminationCondition.doc)
        .value("kEventHandlerFailed", Enum::kEventHandlerFailed,
            enum_doc.kEventHandlerFailed.doc);

    cls  // BR
         // TODO(eric.cousineau): Bind setter methods.
        .def("FormatMessage", &Class::FormatMessage, cls_doc.FormatMessage.doc)
        .def("succeeded", &Class::succeeded, cls_doc.succeeded.doc)
        .def("boundary_time", &Class::boundary_time, cls_doc.boundary_time.doc)
        .def("return_time", &Class::return_time, cls_doc.return_time.doc)
        .def("reason", &Class::reason, cls_doc.reason.doc)
        .def("system", &Class::system, py_rvp::reference, cls_doc.system.doc)
        .def("message", &Class::message, cls_doc.message.doc)
        .def("IsIdenticalStatus", &Class::IsIdenticalStatus, py::arg("other"),
            cls_doc.IsIdenticalStatus.doc);
  }

  {
    constexpr auto& cls_doc = pydrake_doc.drake.systems.InitializeParams;
    using Class = InitializeParams;
    py::class_<Class> cls(m, "InitializeParams", cls_doc.doc);
    cls  // BR
        .def(ParamInit<Class>());
    DefAttributesUsingSerialize(&cls, cls_doc);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }

  auto bind_scalar_types = [m](auto dummy) {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using T = decltype(dummy);
    DefineTemplateClassWithDefault<IntegratorBase<T>>(
        m, "IntegratorBase", GetPyParam<T>(), doc.IntegratorBase.doc)
        .def("set_fixed_step_mode", &IntegratorBase<T>::set_fixed_step_mode,
            doc.IntegratorBase.set_fixed_step_mode.doc)
        .def("get_fixed_step_mode", &IntegratorBase<T>::get_fixed_step_mode,
            doc.IntegratorBase.get_fixed_step_mode.doc)
        .def("set_target_accuracy", &IntegratorBase<T>::set_target_accuracy,
            doc.IntegratorBase.set_target_accuracy.doc)
        .def("get_target_accuracy", &IntegratorBase<T>::get_target_accuracy,
            doc.IntegratorBase.get_target_accuracy.doc)
        .def("set_maximum_step_size", &IntegratorBase<T>::set_maximum_step_size,
            doc.IntegratorBase.set_maximum_step_size.doc)
        .def("get_maximum_step_size", &IntegratorBase<T>::get_maximum_step_size,
            doc.IntegratorBase.get_maximum_step_size.doc)
        .def("set_requested_minimum_step_size",
            &IntegratorBase<T>::set_requested_minimum_step_size,
            doc.IntegratorBase.set_requested_minimum_step_size.doc)
        .def("get_requested_minimum_step_size",
            &IntegratorBase<T>::get_requested_minimum_step_size,
            doc.IntegratorBase.get_requested_minimum_step_size.doc)
        .def("set_throw_on_minimum_step_size_violation",
            &IntegratorBase<T>::set_throw_on_minimum_step_size_violation,
            doc.IntegratorBase.set_throw_on_minimum_step_size_violation.doc)
        .def("get_throw_on_minimum_step_size_violation",
            &IntegratorBase<T>::get_throw_on_minimum_step_size_violation,
            doc.IntegratorBase.get_throw_on_minimum_step_size_violation.doc)
        .def("StartDenseIntegration", &IntegratorBase<T>::StartDenseIntegration,
            doc.IntegratorBase.StartDenseIntegration.doc)
        .def("get_dense_output", &IntegratorBase<T>::get_dense_output,
            py_rvp::reference_internal, doc.IntegratorBase.get_dense_output.doc)
        .def("StopDenseIntegration", &IntegratorBase<T>::StopDenseIntegration,
            doc.IntegratorBase.StopDenseIntegration.doc);

    DefineTemplateClassWithDefault<RungeKutta2Integrator<T>, IntegratorBase<T>>(
        m, "RungeKutta2Integrator", GetPyParam<T>(),
        doc.RungeKutta2Integrator.doc)
        .def(py::init<const System<T>&, const T&, Context<T>*>(),
            py::arg("system"), py::arg("max_step_size"),
            py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, reference: `self` keeps `context` alive.
            py::keep_alive<1, 4>(), doc.RungeKutta2Integrator.ctor.doc);
  };
  type_visit(bind_scalar_types, CommonScalarPack{});

  auto bind_nonsymbolic_scalar_types = [&m](auto dummy) {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using T = decltype(dummy);

    DefineTemplateClassWithDefault<RungeKutta3Integrator<T>, IntegratorBase<T>>(
        m, "RungeKutta3Integrator", GetPyParam<T>(),
        doc.RungeKutta3Integrator.doc)
        .def(py::init<const System<T>&, Context<T>*>(), py::arg("system"),
            py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, reference: `self` keeps `context` alive.
            py::keep_alive<1, 3>(), doc.RungeKutta3Integrator.ctor.doc);

    auto cls = DefineTemplateClassWithDefault<Simulator<T>>(
        m, "Simulator", GetPyParam<T>(), doc.Simulator.doc);
    cls  // BR
        .def(py::init<const System<T>&, unique_ptr<Context<T>>>(),
            py::arg("system"), py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, ownership: `context` keeps `self` alive.
            py::keep_alive<3, 1>(), doc.Simulator.ctor.doc)
        .def("Initialize", &Simulator<T>::Initialize,
            doc.Simulator.Initialize.doc,
            py::arg("params") = InitializeParams{})
        .def("AdvanceTo", &Simulator<T>::AdvanceTo, py::arg("boundary_time"),
            doc.Simulator.AdvanceTo.doc)
        .def("AdvancePendingEvents", &Simulator<T>::AdvancePendingEvents,
            doc.Simulator.AdvancePendingEvents.doc)
        .def("set_monitor", WrapCallbacks(&Simulator<T>::set_monitor),
            py::arg("monitor"), doc.Simulator.set_monitor.doc)
        .def("clear_monitor", &Simulator<T>::clear_monitor,
            doc.Simulator.clear_monitor.doc)
        .def("get_monitor", &Simulator<T>::get_monitor,
            doc.Simulator.get_monitor.doc)
        .def("get_context", &Simulator<T>::get_context,
            py_rvp::reference_internal, doc.Simulator.get_context.doc)
        .def("get_integrator", &Simulator<T>::get_integrator,
            py_rvp::reference_internal, doc.Simulator.get_integrator.doc)
        .def("get_mutable_integrator", &Simulator<T>::get_mutable_integrator,
            py_rvp::reference_internal,
            doc.Simulator.get_mutable_integrator.doc)
        .def("get_mutable_context", &Simulator<T>::get_mutable_context,
            py_rvp::reference_internal, doc.Simulator.get_mutable_context.doc)
        .def("has_context", &Simulator<T>::has_context,
            doc.Simulator.has_context.doc)
        .def("reset_context", &Simulator<T>::reset_context, py::arg("context"),
            // Keep alive, ownership: `context` keeps `self` alive.
            py::keep_alive<2, 1>(), doc.Simulator.reset_context.doc)
        // TODO(eric.cousineau): Bind `release_context` once some form of the
        // PR RobotLocomotion/pybind11#33 lands. Presently, it fails.
        .def("set_publish_every_time_step",
            &Simulator<T>::set_publish_every_time_step, py::arg("publish"),
            doc.Simulator.set_publish_every_time_step.doc)
        .def("set_publish_at_initialization",
            &Simulator<T>::set_publish_at_initialization, py::arg("publish"),
            doc.Simulator.set_publish_at_initialization.doc)
        .def("set_target_realtime_rate",
            &Simulator<T>::set_target_realtime_rate, py::arg("realtime_rate"),
            doc.Simulator.set_target_realtime_rate.doc)
        .def("get_target_realtime_rate",
            &Simulator<T>::get_target_realtime_rate,
            doc.Simulator.get_target_realtime_rate.doc)
        .def("get_actual_realtime_rate",
            &Simulator<T>::get_actual_realtime_rate,
            doc.Simulator.get_actual_realtime_rate.doc)
        .def("ResetStatistics", &Simulator<T>::ResetStatistics,
            doc.Simulator.ResetStatistics.doc)
        .def("get_system", &Simulator<T>::get_system, py_rvp::reference,
            doc.Simulator.get_system.doc);

    m  // BR
        .def("ApplySimulatorConfig",
            py::overload_cast<const SimulatorConfig&,
                drake::systems::Simulator<T>*>(&ApplySimulatorConfig<T>),
            py::arg("config"), py::arg("simulator"),
            pydrake_doc.drake.systems.ApplySimulatorConfig.doc_config_sim)
        .def("ExtractSimulatorConfig", &ExtractSimulatorConfig<T>,
            py::arg("simulator"),
            pydrake_doc.drake.systems.ExtractSimulatorConfig.doc);
    m  // BR
        .def("ApplySimulatorConfig",
            WrapDeprecated(
                pydrake_doc.drake.systems.ApplySimulatorConfig.doc_deprecated,
                [](drake::systems::Simulator<T>* simulator,
                    const SimulatorConfig& config) {
                  ApplySimulatorConfig(config, simulator);
                }),
            py::arg("simulator"), py::arg("config"),
            pydrake_doc.drake.systems.ApplySimulatorConfig.doc_deprecated);
  };
  type_visit(bind_nonsymbolic_scalar_types, NonSymbolicScalarPack{});

  // Simulator Flags
  m  // BR
      .def(
          "ResetIntegratorFromFlags",
          [](Simulator<double>* simulator, const std::string& scheme,
              const double& max_step_size) {
            IntegratorBase<double>& result =
                ResetIntegratorFromFlags(simulator, scheme, max_step_size);
            return &result;
          },
          py::arg("simulator"), py::arg("scheme"), py::arg("max_step_size"),
          py_rvp::reference,
          // Keep alive, reference: `return` keeps `simulator` alive.
          py::keep_alive<0, 1>(),
          pydrake_doc.drake.systems.ResetIntegratorFromFlags.doc)
      .def(
          "ResetIntegratorFromFlags",
          [](Simulator<AutoDiffXd>* simulator, const std::string& scheme,
              const AutoDiffXd& max_step_size) {
            IntegratorBase<AutoDiffXd>& result =
                ResetIntegratorFromFlags(simulator, scheme, max_step_size);
            return &result;
          },
          py::arg("simulator"), py::arg("scheme"), py::arg("max_step_size"),
          py_rvp::reference,
          // Keep alive, reference: `return` keeps `simulator` alive.
          py::keep_alive<0, 1>(),
          pydrake_doc.drake.systems.ResetIntegratorFromFlags.doc)
      .def("GetIntegrationSchemes", &GetIntegrationSchemes,
          pydrake_doc.drake.systems.GetIntegrationSchemes.doc);

  // Print Simulator Statistics
  m  // BR
      .def("PrintSimulatorStatistics", &PrintSimulatorStatistics<double>,
          pydrake_doc.drake.systems.PrintSimulatorStatistics.doc)
      .def("PrintSimulatorStatistics", &PrintSimulatorStatistics<AutoDiffXd>,
          pydrake_doc.drake.systems.PrintSimulatorStatistics.doc);

  // Monte Carlo Testing
  {
    // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
    using namespace drake::systems::analysis;
    constexpr auto& doc = pydrake_doc.drake.systems.analysis;

    m.def("RandomSimulation",
        WrapCallbacks([](const SimulatorFactory make_simulator,
                          const ScalarSystemFunction& output, double final_time,
                          RandomGenerator* generator) -> double {
          return RandomSimulation(
              make_simulator, output, final_time, generator);
        }),
        py::arg("make_simulator"), py::arg("output"), py::arg("final_time"),
        py::arg("generator"), doc.RandomSimulation.doc);

    py::class_<RandomSimulationResult>(
        m, "RandomSimulationResult", doc.RandomSimulationResult.doc)
        .def_readwrite("output", &RandomSimulationResult::output,
            doc.RandomSimulationResult.output.doc)
        .def_readwrite("generator_snapshot",
            &RandomSimulationResult::generator_snapshot,
            doc.RandomSimulationResult.generator_snapshot.doc);

    // Note: parallel simulation must be disabled in the binding via
    // num_parallel_executions=kNoConcurrency, since parallel execution of
    // Python systems in multiple threads is not supported.
    m.def("MonteCarloSimulation",
        WrapCallbacks([](const SimulatorFactory make_simulator,
                          const ScalarSystemFunction& output, double final_time,
                          int num_samples, RandomGenerator* generator)
                          -> std::vector<RandomSimulationResult> {
          return MonteCarloSimulation(make_simulator, output, final_time,
              num_samples, generator, kNoConcurrency);
        }),
        py::arg("make_simulator"), py::arg("output"), py::arg("final_time"),
        py::arg("num_samples"), py::arg("generator"),
        doc.MonteCarloSimulation.doc);

    py::class_<RegionOfAttractionOptions>(
        m, "RegionOfAttractionOptions", doc.RegionOfAttractionOptions.doc)
        .def(py::init<>(), doc.RegionOfAttractionOptions.ctor.doc)
        .def_readwrite("lyapunov_candidate",
            &RegionOfAttractionOptions::lyapunov_candidate,
            doc.RegionOfAttractionOptions.lyapunov_candidate.doc)
        .def_readwrite("state_variables",
            &RegionOfAttractionOptions::state_variables,
            // dtype = object arrays must be copied, and cannot be referenced.
            py_rvp::copy, doc.RegionOfAttractionOptions.state_variables.doc)
        .def_readwrite("use_implicit_dynamics",
            &RegionOfAttractionOptions::use_implicit_dynamics,
            doc.RegionOfAttractionOptions.use_implicit_dynamics.doc)
        .def("__repr__", [](const RegionOfAttractionOptions& self) {
          return py::str(
              "RegionOfAttractionOptions("
              "lyapunov_candidate={}, "
              "state_variables={}, "
              "use_implicit_dynamics={})")
              .format(self.lyapunov_candidate, self.state_variables,
                  self.use_implicit_dynamics);
        });

    m.def("RegionOfAttraction", &RegionOfAttraction, py::arg("system"),
        py::arg("context"), py::arg("options") = RegionOfAttractionOptions(),
        doc.RegionOfAttraction.doc);
  }

  {
    constexpr auto& cls_doc = pydrake_doc.drake.systems.analysis.ControlBarrier;
    using Class = analysis::ControlBarrier;
    auto control_barrier =
        py::class_<Class>(m, "ControlBarrier", cls_doc.doc)
            .def(
                py::init<const Eigen::Ref<const VectorX<symbolic::Polynomial>>&,
                    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>&,
                    std::optional<symbolic::Polynomial>,
                    const Eigen::Ref<const VectorX<symbolic::Variable>>&,
                    double, double, std::vector<VectorX<symbolic::Polynomial>>,
                    const Eigen::Ref<const Eigen::MatrixXd>&,
                    const Eigen::Ref<const VectorX<symbolic::Polynomial>>&>(),
                py::arg("f"), py::arg("G"), py::arg("dynamics_denominator"),
                py::arg("x"), py::arg("beta_minus"), py::arg("beta_plus"),
                py::arg("unsafe_regions"), py::arg("u_vertices"),
                py::arg("state_eq_constraints"), cls_doc.ctor.doc)
            .def(
                "AddControlBarrierConstraint",
                [](const analysis::ControlBarrier& self,
                    solvers::MathematicalProgram* prog,
                    const symbolic::Polynomial& lambda0,
                    const std::optional<symbolic::Polynomial>& lambda1,
                    const VectorX<symbolic::Polynomial>& l,
                    const VectorX<symbolic::Polynomial>&
                        state_constraints_lagrangian,
                    const symbolic::Polynomial& h, double deriv_eps,
                    const std::optional<symbolic::Polynomial>& a) {
                  symbolic::Polynomial hdot_poly;
                  VectorX<symbolic::Monomial> monomials;
                  MatrixX<symbolic::Variable> gram;
                  self.AddControlBarrierConstraint(prog, lambda0, lambda1, l,
                      state_constraints_lagrangian, h, deriv_eps, a, &hdot_poly,
                      &monomials, &gram);
                  return std::make_tuple(hdot_poly, monomials, gram);
                },
                py::arg("prog"), py::arg("lambda0"), py::arg("lambda1"),
                py::arg("l"), py::arg("state_constraints_lagrangian"),
                py::arg("h"), py::arg("deriv_eps"), py::arg("a"),
                cls_doc.AddControlBarrierConstraint.doc);

    py::class_<Class::LagrangianReturn>(control_barrier, "LagrangianReturn")
        .def(
            "prog",
            [](const Class::LagrangianReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("lambda0", &Class::LagrangianReturn::lambda0)
        .def_readonly("lambda0_gram", &Class::LagrangianReturn::lambda0_gram)
        .def_readonly("lambda1", &Class::LagrangianReturn::lambda1)
        .def_readonly("lambda1_gram", &Class::LagrangianReturn::lambda1_gram)
        .def_readonly("l", &Class::LagrangianReturn::l)
        .def_readonly("l_grams", &Class::LagrangianReturn::l_grams)
        .def_readonly("state_constraints_lagrangian",
            &Class::LagrangianReturn::state_constraints_lagrangian)
        .def_readonly("a", &Class::LagrangianReturn::a)
        .def_readonly("a_gram", &Class::LagrangianReturn::a_gram);

    control_barrier.def("ConstructLagrangianProgram",
        &Class::ConstructLagrangianProgram, py::arg("h"), py::arg("deriv_eps"),
        py::arg("lambda0_degree"), py::arg("lambda1_degree"),
        py::arg("l_degrees"), py::arg("state_constraints_lagrangian_degrees"),
        py::arg("a_degree"), cls_doc.ConstructLagrangianProgram.doc);

    py::class_<Class::UnsafeReturn>(control_barrier, "UnsafeReturn")
        .def(
            "prog",
            [](const Class::UnsafeReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("t", &Class::UnsafeReturn::t)
        .def_readonly("t_gram", &Class::UnsafeReturn::t_gram)
        .def_readonly("s", &Class::UnsafeReturn::s)
        .def_readonly("s_grams", &Class::UnsafeReturn::s_grams)
        .def_readonly("state_constraints_lagrangian",
            &Class::UnsafeReturn::state_constraints_lagrangian)
        .def_readonly("a", &Class::UnsafeReturn::a)
        .def_readonly("a_gram", &Class::UnsafeReturn::a_gram)
        .def_readonly("sos_poly", &Class::UnsafeReturn::sos_poly)
        .def_readonly("sos_poly_gram", &Class::UnsafeReturn::sos_poly_gram);

    control_barrier.def("ConstructUnsafeRegionProgram",
        &Class::ConstructUnsafeRegionProgram, py::arg("h"),
        py::arg("region_index"), py::arg("t_degree"), py::arg("s_degrees"),
        py::arg("state_constraints_lagrangian_degrees"), py::arg("a_degree"),
        cls_doc.ConstructUnsafeRegionProgram.doc);

    py::class_<Class::BarrierReturn>(control_barrier, "BarrierReturn")
        .def(
            "prog",
            [](const Class::BarrierReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("h", &Class::BarrierReturn::h)
        .def_readonly("hdot_sos", &Class::BarrierReturn::hdot_sos)
        .def_readonly("hdot_sos_gram", &Class::BarrierReturn::hdot_sos_gram)
        .def_readonly("hdot_state_constraints_lagrangian",
            &Class::BarrierReturn::hdot_state_constraints_lagrangian)
        .def_readonly("hdot_a", &Class::BarrierReturn::hdot_a)
        .def_readonly("hdot_a_gram", &Class::BarrierReturn::hdot_a_gram)
        .def_readonly("s", &Class::BarrierReturn::s)
        .def_readonly("s_grams", &Class::BarrierReturn::s_grams)
        .def_readonly(
            "unsafe_sos_polys", &Class::BarrierReturn::unsafe_sos_polys)
        .def_readonly("unsafe_sos_poly_grams",
            &Class::BarrierReturn::unsafe_sos_poly_grams)
        .def_readonly("unsafe_state_constraints_lagrangian",
            &Class::BarrierReturn::unsafe_state_constraints_lagrangian)
        .def_readonly("unsafe_a", &Class::BarrierReturn::unsafe_a)
        .def_readonly("unsafe_a_gram", &Class::BarrierReturn::unsafe_a_gram);

    control_barrier.def("ConstructBarrierProgram",
        &Class::ConstructBarrierProgram, py::arg("lambda0"), py::arg("lambda1"),
        py::arg("l"), py::arg("hdot_state_constraints_lagrangian_degrees"),
        py::arg("hdot_a_degree"), py::arg("t"),
        py::arg("unsafe_state_constraints_lagrangian_degrees"),
        py::arg("h_degree"), py::arg("deriv_eps"), py::arg("s_degrees"),
        py::arg("unsafe_a_degrees"), cls_doc.ConstructBarrierProgram.doc);
  }
}

}  // namespace pydrake
}  // namespace drake
