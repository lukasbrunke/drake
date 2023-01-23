import typing
import numpy as np
import pdb

from pydrake.geometry import (AddContactMaterial, CollisionFilterDeclaration,
                              Cylinder, GeometrySet, Meshcat,
                              MeshcatVisualizer, MeshcatVisualizerParams,
                              ProximityProperties, Role, SceneGraph)
from pydrake.geometry.optimization import HPolyhedron
from pydrake.geometry.optimization_dev import (CspaceFreePolytope,
                                               SeparatingPlane,
                                               SeparatingPlaneOrder)
from pydrake.common import (
    FindResourceOrThrow, )
from pydrake.systems.framework import (Context, Diagram, DiagramBuilder)
from pydrake.multibody.plant import (AddMultibodyPlantSceneGraph,
                                     CoulombFriction, MultibodyPlant)
from pydrake.multibody.tree import (ModelInstanceIndex)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import (RigidTransform, RollPitchYaw)
from pydrake.solvers import mathematicalprogram as mp


class UrDiagram:
    diagram: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    meshcat: Meshcat
    visualizer: MeshcatVisualizer
    ur_instances: typing.List[ModelInstanceIndex]
    gripper_instances: typing.List[ModelInstanceIndex]

    def __init__(self, num_ur: int, weld_wrist: bool, add_shelf: bool,
                 add_gripper: bool):
        self.meshcat = Meshcat()
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, 0.0)
        parser = Parser(self.plant)
        ur_file_name = "drake/manipulation/models/ur3e/" + (
            "ur3e_cylinder_weld_wrist.urdf"
            if weld_wrist else "ur3e_cylinder_revolute_wrist.urdf")
        ur_file_path = FindResourceOrThrow(ur_file_name)
        self.ur_instances = []
        self.gripper_instances = []
        for ur_count in range(num_ur):
            ur_instance = parser.AddModelFromFile(ur_file_path,
                                                  f"ur{ur_count}")
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.plant.GetFrameByName("ur_base_link", ur_instance),
                RigidTransform(np.array([0, ur_count * 0.6, 0])))
            self.ur_instances.append(ur_instance)
            if add_gripper:
                gripper_file_path = FindResourceOrThrow(
                    "drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf"
                )
                gripper_instance = parser.AddModelFromFile(
                    gripper_file_path, f"schunk{ur_count}")
                self.gripper_instances.append(gripper_instance)
                self.plant.WeldFrames(
                    self.plant.GetBodyByName("ur_ee_link",
                                             ur_instance).body_frame(),
                    self.plant.GetBodyByName("body",
                                             gripper_instance).body_frame(),
                    RigidTransform(RollPitchYaw(0, 0, -np.pi / 2),
                                   np.array([0.06, 0, 0])))

        if add_shelf:
            shelf_file_path = FindResourceOrThrow(
                "drake/geometry/optimization/dev/models/shelves.sdf")
            shelf_instance = parser.AddModelFromFile(shelf_file_path,
                                                     "shelves")
            shelf_body = self.plant.GetBodyByName("shelves_body",
                                                  shelf_instance)
            shelf_frame = self.plant.GetFrameByName("shelves_body",
                                                    shelf_instance)
            X_WShelf = RigidTransform(np.array([0.6, 0, 0.4]))
            self.plant.WeldFrames(self.plant.world_frame(), shelf_frame,
                                  X_WShelf)

            proximity_properties = ProximityProperties()
            AddContactMaterial(dissipation=0.1,
                               point_stiffness=250.0,
                               friction=CoulombFriction(0.9, 0.5),
                               properties=proximity_properties)
            shelf_cylinder = self.plant.RegisterCollisionGeometry(
                shelf_body, RigidTransform(np.array([0, 0, -0.05])),
                Cylinder(0.02, 0.15), "shelf_cylinder", proximity_properties)

        self.plant.Finalize()

        inspector = self.scene_graph.model_inspector()
        for ur_instance in self.ur_instances:
            ur_geometries = GeometrySet()
            for body_index in self.plant.GetBodyIndices(ur_instance):
                body_geometries = inspector.GetGeometries(
                    self.plant.GetBodyFrameIdOrThrow(body_index))
                ur_geometries.Add(body_geometries)
            self.scene_graph.collision_filter_manager().Apply(
                CollisionFilterDeclaration().ExcludeWithin(ur_geometries))

        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            builder, self.scene_graph, self.meshcat, meshcat_params)
        print(self.meshcat.web_url())
        self.diagram = builder.Build()


def save_result(search_result: CspaceFreePolytope.SearchResult,
                file_path: str):
    np.savez(file_path, search_result.C, search_result.d)


def find_ur_shelf_posture(plant: MultibodyPlant,
                          gripper_instance: ModelInstanceIndex,
                          plant_context: Context):
    ik = InverseKinematics(plant, plant_context)
    ee_frame = plant.GetFrameByName("ur_ee_link")
    gripper_frame = plant.GetBodyByName("body", gripper_instance).body_frame()
    shelf_frame = plant.GetFrameByName("shelves_body")
    ik.AddPositionConstraint(gripper_frame, np.zeros((3, )), shelf_frame,
                             np.array([-0.15, -0., -0.2]),
                             np.array([0.05, 0., 0.2]))
    ik.AddPositionConstraint(gripper_frame, np.array([0, 0.028, 0]),
                             shelf_frame, np.array([-0.05, -0.02, -0.15]),
                             np.array([0.05, 0.02, 0.12]))
    #ik.AddAngleBetweenVectorsConstraint(gripper_frame, np.array([1, 0, 0.]),
    #                                    plant.world_frame(),
    #                                    np.array([0, 0, 1]), 0.45 * np.pi,
    #                                    0.55 * np.pi)
    #right_finger = plant.GetBodyByName("right_finger", gripper_instance)
    #ik.AddPositionConstraint(right_finger.body_frame(), np.array([0., 0., 0.]),
    #                         shelf_frame, np.array([-0.15, 0.04, -0.15]),
    #                         np.array([0.15, 0.1, 0.2]))
    #left_finger = plant.GetBodyByName("left_finger", gripper_instance)
    #ik.AddPositionConstraint(left_finger.body_frame(), np.array([0., 0., 0.]),
    #                         shelf_frame, np.array([-0.15, -0.1, -0.15]),
    #                         np.array([0.15, -0.01, 0.2]))
    ik.AddMinimumDistanceConstraint(0.015)

    q_init = np.array([-0.4, 0.7, -0., 0.7, 0.5, 0])
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q_init)
    result = mp.Solve(ik.prog(), q_init, None)
    if not result.is_success():
        raise Warning("Cannot find the posture")
    print(result.GetSolution(ik.q()))
    return result.GetSolution(ik.q())


def setup_ur_shelf_cspace_polytope(
        s_init: np.ndarray) -> (np.ndarray, np.ndarray):
    C = np.array((12, 6))
    d = np.array((12, ))
    C = np.array([[1.5, 0.1, 0.2, -0.1, -0.2, 0.4],
                  [-2.1, 0.2, -0.1, 0.4, -0.3, 0.1],
                  [0.1, 2.5, 0.3, -0.3, 0.2, -0.2],
                  [-0.3, -2.1, 0.2, 0.3, -0.4, 0.1],
                  [0.1, 0.2, 3.2, -0.1, 0.3, -0.2],
                  [0.2, -0.1, -2.5, 0.2, 0.3, -0.1],
                  [0.2, 0.1, 1.2, 3.2, 0.2, 0.3],
                  [-0.2, 0.3, -0.4, -4.1, 0.4, -0.2],
                  [0.4, 0.2, 0.5, -0.3, 3.2, -0.2],
                  [-0.1, -0.5, 0.2, -0.5, -2.9, 0.3],
                  [0.1, 1.2, 0.4, 1.5, -0.4, 2.3],
                  [0.2, -0.3, -1.5, 0.2, 2.1, -3.4]])
    d = np.array([
        0.1, 0.05, 0.1, 0.2, 0.05, 0.15, 0.2, 0.1, 0.4, 0.1, 0.2, 0.2
    ]) + C @ s_init

    hpolyhedron = HPolyhedron(C, d)
    assert (hpolyhedron.IsBounded())
    return C, d


def search_ur_shelf_cspace_polytope(weld_wrist: bool, with_gripper: bool):
    ur_diagram = UrDiagram(num_ur=1,
                           weld_wrist=weld_wrist,
                           add_shelf=True,
                           add_gripper=with_gripper)
    diagram_context = ur_diagram.diagram.CreateDefaultContext()
    plant_context = ur_diagram.plant.GetMyMutableContextFromRoot(
        diagram_context)
    q_init = find_ur_shelf_posture(ur_diagram.plant,
                                   ur_diagram.gripper_instances[0],
                                   plant_context)
    ur_diagram.plant.SetPositions(plant_context, q_init)
    ur_diagram.diagram.ForcedPublish(diagram_context)
    pdb.set_trace()
    q_star = np.zeros((6, ))
    cspace_free_polytope_options = CspaceFreePolytope.Options()
    cspace_free_polytope_options.with_cross_y = False
    cspace_free_polytope = CspaceFreePolytope(ur_diagram.plant,
                                              ur_diagram.scene_graph,
                                              SeparatingPlaneOrder.kAffine,
                                              q_star,
                                              cspace_free_polytope_options)

    s_init = cspace_free_polytope.rational_forward_kin().ComputeSValue(
        q_init, q_star)

    binary_search = False
    binary_search_data = "/home/hongkaidai/Dropbox/c_iris_data/ur/ur_shelf_binary_search.npz"
    ignored_collision_pairs = set()
    if binary_search:
        C_init, d_init = setup_ur_shelf_cspace_polytope(s_init)

        binary_search_options = CspaceFreePolytope.BinarySearchOptions()
        binary_search_options.scale_max = 0.3
        binary_search_options.scale_min = 0.05
        binary_search_options.max_iter = 3
        binary_search_options.find_lagrangian_options.verbose = True
        binary_search_options.find_lagrangian_options.num_threads = -1

        binary_search_result = cspace_free_polytope.BinarySearch(
            ignored_collision_pairs, C_init, d_init, s_init,
            binary_search_options)
        np.savez(binary_search_data,
                 C=binary_search_result.C,
                 d=binary_search_result.d,
                 s_init=s_init)
        C_start = binary_search_result.C
        d_start = binary_search_result.d
        pdb.set_trace()
    else:
        binary_search_file = np.load(binary_search_data)
        C_start = binary_search_file["C"]
        d_start = binary_search_file["d"]
        s_init = binary_search_file["s_init"]

    bilinear_alternation_options = CspaceFreePolytope.BilinearAlternationOptions(
    )
    bilinear_alternation_options.find_lagrangian_options.num_threads = -1
    bilinear_alternation_options.convergence_tol = 1E-14
    bilinear_alternation_options.find_polytope_options.s_inner_pts = s_init.reshape(
        (-1, 1))
    bilinear_alternation_options.find_polytope_options.solver_options = \
        mp.SolverOptions()
    bilinear_alternation_options.find_polytope_options.solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    bilinear_alternation_options.find_polytope_options.backoff_scale = 0.2
    # bilinear_alternation_options.find_polytope_options.search_s_bounds_lagrangians = False
    bilinear_alternation_result = cspace_free_polytope.SearchWithBilinearAlternation(
        ignored_collision_pairs, C_start, d_start,
        bilinear_alternation_options)
    pdb.set_trace()

    pass


if __name__ == "__main__":
    search_ur_shelf_cspace_polytope(weld_wrist=False, with_gripper=True)
