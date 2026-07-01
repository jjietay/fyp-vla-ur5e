import socket
import time

class DashboardClient:
    def __init__(self, ip: str, port: int = 29999):
        self.sock = socket.create_connection((ip, port), timeout=5)
        self._recv() # consumes the initial "Connected: Universal Robots..." banner

    def send(self, command: str) -> str:
        self.sock.sendall((command + "\n").encode())
        return self._recv()
    
    def _recv(self) -> str:
        return self.sock.recv(1024).decode().strip()
    
    def close(self):
        self.sock.close()


if __name__ == "__main__":
    d = DashboardClient("127.0.0.1")
    try:
        print(d.send("power on"))
        print(d.send("brake release"))
        time.sleep(1)   # wait for the brake to be released
        print(d.send("robotmode"))
    finally:
        d.close()
