from director import lcmUtils
from director import roboturdf
import pickle
import bot_core as lcmbotcore

# receive lcm message and draw data
def receiveMessage(msg):

    robotModel, jointController = roboturdf.loadRobotModel(urdfFile='/home/hongkai/drake-distro/drake/examples/IRB140/urdf/irb_140_shift.urdf', view=view, useConfigFile=False)
    jointController.setPose('my posture', np.zeros(len(jointController.jointNames)))

    folderName = 'my data'

    # remove the folder completely
    om.removeFromObjectModel(om.findObjectByName(folderName))

    #create a folder
    folder = om.getOrCreateContainer(folderName)

    # unpack message
    data = pickle.loads(msg.data)

    d1 = DebugData()
    d2 = DebugData()
    d3 = DebugData()
    
    file = open('/home/hongkai/drake-distro/ik_output21_1.txt','r')

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
        elif line.startswith("nonlinear_ik_status:"):
            nonlinear_ik_status_str = line.split()
            nonlinear_ik_status = int(nonlinear_ik_status_str[1])
        elif line.startswith("global_ik_status:"):
            global_ik_status_str = line.split()
            global_ik_status = int(global_ik_status_str[1])
        elif line.startswith("q_nonlinear_ik_resolve:"):
            if (analytical_ik_status == 0 or nonlinear_ik_status == 0) and (global_ik_status == 0):
                # Analytical IK and global IK both find solution
                d1.addSphere(pos, radius = 0.006, color = [0, 1, 0])
            elif (analytical_ik_status == -2 and global_ik_status == -2):
                d2.addSphere(pos, radius = 0.006, color = [0, 0, 1])
            elif (analytical_ik_status == -2 and global_ik_status == 0):
                d3.addSphere(pos, radius = 0.006, color = [1, 0, 0])
        line_number =  line_number + 1

    
    vis.showPolyData(d1.getPolyData(), 'reachable', parent=folder, colorByName='RGB255')
    vis.showPolyData(d2.getPolyData(), 'unreachable', parent=folder, colorByName='RGB255')
    vis.showPolyData(d3.getPolyData(), 'relaxation', parent=folder, colorByName='RGB255')

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
