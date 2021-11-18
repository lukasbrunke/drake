import sys
import os
import time
import numpy as np

import pydrake
from pydrake.all import BsplineTrajectoryThroughUnionOfHPolyhedra, IrisInConfigurationSpace, IrisOptions
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.optimization import CalcGridPointsOptions, Toppra
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import RevoluteJoint
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.solvers.mosek import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import Variable, Expression, RotationMatrix
from pydrake.all import MultibodyPositionToGeometryPose, ConnectMeshcatVisualizer, Role, Sphere
from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
    Role, Sphere,
    Iris, IrisOptions, MakeIrisObstacles, Variable
)
from pydrake.all import (
    eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff,
)

import pydrake.symbolic as sym

from meshcat import Visualizer

from pydrake.all import RationalFunction
from pydrake.all import RationalForwardKinematics
import unittest


def construct_iiwa():
    simple_collision = True
    # gripper_welded = True

    #vis = Visualizer(zmq_url=zmq_url)
    #vis.delete()
    #display(vis.jupyter_cell())

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.package_map().Add( "wsg_50_description", os.path.dirname(FindResourceOrThrow(
                "drake/manipulation/models/wsg_50_description/package.xml")))

    directives_file = FindResourceOrThrow("drake/sandbox/planar_iiwa_simple_collision_welded_gripper.yaml") \
        if simple_collision else FindResourceOrThrow("drake/sandbox/planar_iiwa_dense_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)

    q0 = [-0.2, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(models[0].model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    plant.Finalize()

    #visualizer = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url, delete_prefix_on_load=False)

    diagram = builder.Build()
    #visualizer.load()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    diagram.Publish(context)
    return diagram, plant, scene_graph

class Test(unittest.TestCase):
    def test(self):
        _, plant, _ = construct_iiwa()
        forward_kin = RationalForwardKinematics(plant)
        for i in range(forward_kin.t().shape[0]):
            print(f"{forward_kin.t()[i].get_name()} {forward_kin.t()[i].get_id()}")
        q_star = np.zeros(forward_kin.t().shape[0])
        dummy_var = pydrake.symbolic.Variable("dummy")
        print(f"dummy_var id: {dummy_var.get_id()}")
        for i in range(forward_kin.t().shape[0]):
            self.assertGreater(dummy_var.get_id(), forward_kin.t()[i].get_id())
