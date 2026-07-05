## To run URSim
URSIM=~/Documents/NTU/Y4S1/FYP/ursim/URSim_Linux-5.25.2.130406/ursim-5.25.2.130406

cd "$URSIM"

./start-ursim.sh UR5e

## Unrestricted starting pose values
pose = [0.44489, -0.24078, -0.23421, 3.075, 0.679, -0.002]

## For pytest
uv run pytest --name-of-file.py

## Run MuJoCo
XDG_SESSION_TYPE=x11 uv run python -m mujoco.viewer --mjcf assets/mujoco/ur5e/ur5e.xml


## Start MuJoCo Server
XDG_SESSION_TYPE=x11 uv run python -m fyp.sim.sim_server

## Start Client
uv run python

from fyp.sim.sim_client import SimClient
c = SimClient()
c.get_state()
c.move_joints([-1.0, -1.5, 1.5, -1.5, -1.5, 0.0], speed=1.0)