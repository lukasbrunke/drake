from director import lcmUtils
from director import roboturdf
import pickle
import bot_core as lcmbotcore

# receive lcm message and draw data
def receiveMessage(msg):
    drake_path = '/home/hongkai/drake-distro'
    case = 0

    #robotModel, jointController = roboturdf.loadRobotModel(urdfFile=drake_path+'/drake/examples/IRB140/urdf/irb_140_shift.urdf', view=view, useConfigFile=False)
    #jointController.setPose('my posture', np.zeros(len(jointController.jointNames)))

    #ee_model, ee_joint_controller = roboturdf.loadRobotModel(urdfFile=drake_path+'/drake/examples/IRB140/urdf/end_effector.urdf', view=view, useConfigFile=False)
    ee_pose = np.array([-0.5, -0.4, 0.5, 0, 0, 0])
    if case == 0 :
        ee_pose[5] = 0
    elif case == 2:
        ee_pose[5] = 1.57
    #ee_joint_controller.setPose('ee posture', ee_pose)
    folderName = 'my data'

    # remove the folder completely
    om.removeFromObjectModel(om.findObjectByName(folderName))

    #create a folder
    folder = om.getOrCreateContainer(folderName)

    # unpack message
    data = pickle.loads(msg.data)

    d_reachable = DebugData()
    d_unreachable_le_100ms = DebugData()
    d_unreachable_le_1s = DebugData()
    d_unreachable_le_1m = DebugData()
    d_unreachable_ge_1m = DebugData()
    d_relaxation = DebugData()
    d_problem = DebugData()
    
    file = open(drake_path+'/ik_output21_' + str(case) +'.txt','r')

    lines = file.readlines()

    line_number = 0

    while line_number < len(lines):
        line = lines[line_number]
        if line.startswith("position:"):
            line_number = line_number + 1
            pos_str = lines[line_number].split()
            pos = [float(pos_str[0]), float(pos_str[1]), float(pos_str[2])]
        elif line.startswith("analytical_ik_status:"):
            analytical_ik_status_str = line.split()
            analytical_ik_status = int(analytical_ik_status_str[1])
            if case == 2:
                if analytical_ik_status == 0:
                    q_analytical_str = lines[line_number + 2].split()
                    q_analytical = np.zeros(12)
                    for i in range(6):
                        q_analytical[i + 6] = float(q_analytical_str[i])
                    if (pos[0] == 0.5 and pos[1] == 0 and pos[2] == 0.6):
                        jointController.setPose('my posture', q_analytical)
        elif line.startswith("nonlinear_ik_status:"):
            nonlinear_ik_status_str = line.split()
            nonlinear_ik_status = int(nonlinear_ik_status_str[1])
        elif line.startswith("global_ik_status:"):
            global_ik_status_str = line.split()
            global_ik_status = int(global_ik_status_str[1])
        elif line.startswith("global_ik_time:"):
            global_ik_time_str = line.split()
            global_ik_time = float(global_ik_time_str[1])
        elif line.startswith("q_nonlinear_ik_resolve:"):
            if (analytical_ik_status == 0 or nonlinear_ik_status == 1) and (global_ik_status == 0):
                # Analytical IK and global IK both find solution
                d_reachable.addSphere(pos, radius = 0.01, color = [0, 1, 0])
            elif (analytical_ik_status == -2 and global_ik_status == -2):
                if global_ik_time < 0.1:
                    d_unreachable_le_100ms.addSphere(pos, radius = 0.01, color = [0, 0, 1])
                elif global_ik_time < 1:
                    d_unreachable_le_1s.addSphere(pos, radius = 0.01, color = [0, 0, 1])
                elif global_ik_time < 60:
                    d_unreachable_le_1m.addSphere(pos, radius = 0.01, color = [0, 0, 1])
                else:
                    d_unreachable_ge_1m.addSphere(pos, radius = 0.01, color = [0, 0, 1])
            elif (analytical_ik_status == -2 and global_ik_status == 0):
                d_relaxation.addSphere(pos, radius = 0.01, color = [1, 0, 0])
            elif analytical_ik_status == 0 and global_ik_status == -2:
                d_problem.addSphere(pos, radius = 0.01, color = [0, 1, 0])
        line_number =  line_number + 1

    
    vis.showPolyData(d_reachable.getPolyData(), 'reachable', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_unreachable_le_100ms.getPolyData(), 'unreachable<100ms', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_unreachable_le_1s.getPolyData(), 'unreachable<1s', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_unreachable_le_1m.getPolyData(), 'unreachable<1m', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_unreachable_ge_1m.getPolyData(), 'unreachable>1m', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_relaxation.getPolyData(), 'relaxation', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_problem.getPolyData(), 'problem', parent=folder, colorByName='RGB255')

    camera = view.camera()
    camera.SetPosition([-0.7755, -2.1533, 2.8535])
    camera.SetFocalPoint([0.3508, -0.1305, 0.5296])
    view.forceRender()

def publishData():
    data = 1
    msg = lcmbotcore.raw_t()
    msg.data = pickle.dumps(data)
    msg.length = len(msg.data)
    lcmUtils.publish('MY_DATA', msg)


# add an lcm subscriber with a python callback
lcmUtils.addSubscriber('MY_DATA', messageClass=lcmbotcore.raw_t, callback=receiveMessage)

# publish an lcm message
publishData()

# call myTimer.start() to begin publishing
myTimer = TimerCallback(callback = publishData, targetFps = 10)
