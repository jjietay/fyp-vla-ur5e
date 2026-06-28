import rtde_control

rtde_c = rtde_control.RTDEControlInterface(
    "127.0.0.1", 500.0, rtde_control.RTDEControlInterface.FLAG_VERBOSE
)
print("CONNECTED:", rtde_c.isConnected())
rtde_c.moveJ([0, -1.5708, 1.5708, -1.5708, -1.5708, 0], 1.0, 1.0)
rtde_c.stopScript()