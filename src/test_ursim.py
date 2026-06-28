import socket   # gives tcp networking api
import time     # only used to pause briefly before program exits

HOST = "127.0.0.1"  # connect to this machine 127.0.0.1, docker publishes URSim port on this mac
PORT = 30002        # UR Secondary Interface, which accepts URScript sent over TCP

# \n is required because its line based text over the socket
# a is acceleration and v is velocity
# the values in movej is absolute joint configuration, 6 joint angles not the [x,y,z,rx,ry,rz] rotation vector
cmd = "movej([0,-1.57,1.57,0,1.57,0], a=1.0, v=0.5)\n"

# this creates a TCP socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    '''
    AF_INET --> IPv4
    SOCK_STREAM --> TCP
    '''
    
    # opens the TCP connection to URSim at 127.0.0.1:30002
    s.connect((HOST, PORT))
    
    # sockets sends bytes, not python strings, so encode converts them into bytes first
    # sendall is different from send because sendall it will keep sending ALL the bytes or until an error occurs
    s.sendall(cmd.encode("utf-8"))
    
    time.sleep(1)