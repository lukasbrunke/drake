from director import lcmUtils
from director import roboturdf
import pickle
import bot_core as lcmbotcore

# receive lcm message and draw data
def receiveMessage(msg):
    drake_path = '/home/hongkai/drake-distro'
    case = 0

    folderName = 'my data'

    # remove the folder completely
    om.removeFromObjectModel(om.findObjectByName(folderName))

    #create a folder
    folder = om.getOrCreateContainer(folderName)

    # unpack message
    data = pickle.loads(msg.data)

    d_nl_ik_succeed= DebugData()
    d_nl_ik_fail= DebugData()
    
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
        elif line.startswith("nonlinear_ik_status:"):
            nonlinear_ik_status_str = line.split()
            nonlinear_ik_status = int(nonlinear_ik_status_str[1])
        elif line.startswith("q_nonlinear_ik_resolve:"):
            if (analytical_ik_status == 0 and nonlinear_ik_status < 10):
                d_nl_ik_succeed.addSphere(pos, radius = 0.01, color = [0, 1, 0])
            elif(analytical_ik_status == 0 and nonlinear_ik_status > 10):
                d_nl_ik_fail.addSphere(pos, radius = 0.01, color = [1, 0, 0])
        line_number =  line_number + 1

    
    vis.showPolyData(d_nl_ik_succeed.getPolyData(), 'nl_ik_succeed', parent=folder, colorByName='RGB255')
    vis.showPolyData(d_nl_ik_fail.getPolyData(), 'nl_ik_fail', parent=folder, colorByName='RGB255')

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
