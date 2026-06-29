# URSim 5.25.2 LTS: Setup Reference (Ubuntu 24.04)

> URSim and its installer scripts are property of Universal Robots. This document describes patches a user can apply to their own legally-obtained copy of URSim to get it running on a modern Ubuntu stack. It does not redistribute UR's software.

Reference for the working install. URSim was built for Ubuntu 14.04 / JDK 1.8, so getting it running on Ubuntu 24.04 requires several patches. This document captures the final state.

**Tested on:**
- Ubuntu 24.04.1 LTS (Noble Numbat), kernel 6.8.x
- OpenJDK 8 (for URSim) alongside OpenJDK 21 (system default, untouched)
- URSim version: 5.25.2.130406 (Linux non-Docker)
- Hardware: Lenovo Legion Pro 5i 16IAX10H
- Confirmed working: June 2026

## System

- OS: Ubuntu 24.04 (Noble Numbat)
- Default Java: OpenJDK 21 (untouched: URSim is forced to use Java 8 locally)
- Display server: Wayland or Xorg (Xorg recommended if visual glitches persist)

## Parameter

Set this once and the rest of the document is reusable verbatim:

```bash
export URSIM="$HOME/path/to/URSim_Linux-5.25.2.130406/ursim-5.25.2.130406"
```

All paths below are relative to `$URSIM` unless otherwise noted.

## Paths

| What | Where |
|---|---|
| URSim root | `$URSIM` |
| Startup script | `$URSIM/start-ursim.sh` |
| Stop script | `$URSIM/stopurcontrol.sh` |
| Controller binary | `$URSIM/URControl` |
| Controller log (silent crashes go here) | `$URSIM/URControl.log` (runtime-generated) |
| Polyscope log | `$URSIM/polyscope.log` (runtime-generated) |
| Runtime config dir | `$URSIM/.urcontrol/` (created on first launch) |
| Bundled dynamic libs (32-bit) | `$URSIM/dynlibs/` |
| Required symlink | `/usr/local/urcontrol/dynlibs` --> `$URSIM/dynlibs` |
| Java 8 home | `/usr/lib/jvm/java-8-openjdk-amd64` |
| Program/installation files (UR5e) | `$URSIM/programs.UR5/` |
| Active model symlink (created on launch) | `$URSIM/programs` --> `programs.UR5/` (depends on model launched) |

## Dependencies

Manually installed:

- `openjdk-8-jre` : required by URSim's OSGi bundles. System default Java 21 remains untouched; Java 8 is selected per-launch via `JAVA_HOME` in the startup script.
- `libjava3d-jni`, `libjava3d-java` : fixes the `j3dcore-ogl` crash at 100% load.

Auto-pulled by `install.sh` (after patches):

- `libcurl4t64` (modern replacement for the hardcoded `libcurl3`)
- `lib32gcc-s1`, `lib32stdc++6`, `libc6-i386` (32-bit runtime: URControl is a 32-bit binary)
- `libnsl2`, `libvecmath-java`, various fonts
- `curl-dev-ur`, `libxmlrpc-c-ur`, `libxmlrpc-c-dev-ur` (UR's bundled `.deb` packages)
- `runit`, `runit-run`, `sysuser-helper` (daemon manager: URControl runs as a runit service in 5.25.x)

## Modifications made

1. **`install.sh`** : renamed obsolete package names so apt can resolve them:
   - `libcurl3` --> `libcurl4`
   - `lib32gcc1` --> `lib32gcc-s1`

2. **`ursim-dependencies/libxmlrpc-c-ur_1.33.14_amd64.deb`** : control file edited to use modern `lib32gcc-s1` and drop the strict epoch version constraint `(>= 1:4.1.1)`, which dpkg refused to satisfy with Ubuntu 24.04's `lib32gcc-s1` 14.2.0 (epoch 0 < epoch 1 by Debian rules).

3. **`start-ursim.sh`** : prepended three exports immediately after the shebang:
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   export PATH=$JAVA_HOME/bin:$PATH
   export _JAVA_AWT_WM_NONREPARENTING=1
   ```
   The first two force Java 8 (without touching system defaults). The third stops Java's AWT from fighting with modern Linux window managers, reducing fullscreen visual tearing.

4. **`/usr/local/urcontrol/dynlibs`** : symlink created pointing to `$URSIM/dynlibs`. URControl has this path hardcoded in its ELF interpreter : the installer is supposed to create the link but bails out before reaching that step on Ubuntu 24.04.

## Setup procedure (replayable)

```bash
# Set the path (edit to your actual install location)
export URSIM="$HOME/path/to/URSim_Linux-5.25.2.130406/ursim-5.25.2.130406"

# 1. Prerequisites
sudo apt update
sudo apt install -y openjdk-8-jre libjava3d-jni libjava3d-java

# 2. Patch install.sh
cd "$URSIM"
sed -i 's/libcurl3/libcurl4/g; s/lib32gcc1\b/lib32gcc-s1/g' install.sh

# 3. Patch the bundled .deb
cd "$URSIM/ursim-dependencies"
ar x libxmlrpc-c-ur_1.33.14_amd64.deb
mkdir -p extras-control
tar -C extras-control -zxf control.tar.gz
sed -i 's/lib32gcc1 (>= 1:4.1.1)/lib32gcc-s1/g; s/lib32gcc1/lib32gcc-s1/g' extras-control/control
( cd extras-control && tar cfz ../control.tar.gz . )
ar r libxmlrpc-c-ur_1.33.14_amd64.deb control.tar.gz

# 4. Run installer + clean up any partial config
cd "$URSIM"
sudo ./install.sh
sudo apt --fix-broken install -y

# 5. Symlink the installer forgets
sudo mkdir -p /usr/local/urcontrol
sudo ln -sfn "$URSIM/dynlibs" /usr/local/urcontrol/dynlibs

# 6. Force Java 8 in the startup script
sed -i '2i export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64\nexport PATH=$JAVA_HOME/bin:$PATH\nexport _JAVA_AWT_WM_NONREPARENTING=1' start-ursim.sh
```

## Launching

```bash
cd "$URSIM"
./start-ursim.sh UR5e
```

Other model flags: `UR3e`, `UR10e`, `UR16e`, `UR20`, `UR30`. Pick the one that matches the target arm (UR5e for most FYP setups).

The first launch will create `$URSIM/programs` (symlink to the active model's program dir) and `$URSIM/.urcontrol/` (runtime config). It will also create `default.installation` and `default.variables` inside `programs.UR5/` if they don't exist. None of this requires action : it's normal first-run behaviour.

## Stopping cleanly

```bash
./stopurcontrol.sh         # ends URControl gracefully
# or, if hung:
sudo killall -9 URControl java
```

## Connecting from `ur_rtde`

URSim listens on `127.0.0.1` (localhost) by default.

| Port | Purpose |
|---|---|
| 30001 | Primary client interface |
| 30002 | Secondary client interface (URScript over TCP) |
| 30003 | Real-time client interface |
| 30004 | RTDE (used by `ur_rtde`) |
| 29999 | Dashboard server |

Python example:

```python
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

rtde_c = RTDEControlInterface("127.0.0.1")
rtde_r = RTDEReceiveInterface("127.0.0.1")
print(rtde_r.getActualTCPPose())
```

## Harmless warnings to ignore

These all show up in a healthy run. Don't burn time chasing them:

- `URControl: no process found` at the start of `./start-ursim.sh` : script is about to spawn one; this is the "is one already running?" check.
- `./install.sh: line 107: /root/Desktop/ursim-5.25.2.130406.UR3.desktop: No such file or directory` : installer tries to create desktop shortcuts for root, which has no Desktop folder. Doesn't affect the install.
- `usermod: The previous home directory (/home/_runit-log) does not exist or is inaccessible` : `runit` package setup quirk during apt install.
- `Failed to load installation` --> `creating 'default.installation'` : first-run only. Creates defaults.
- `ClassNotFoundException: com.steadystate.css.parser.SACParserCSS21 not found` --> `using the default 'SACParserCSS21' instead` : falls back cleanly.
- `Socket connection to calibration backend failed` : the calibration backend isn't part of the sim. Fine.
- `Dashboard setting programstate to "Stopped" since programName "rtde_control" was not recognized` : Polyscope doesn't know about external RTDE program names. The script is still running; this is just a UI labelling issue.
- `Method used before correct value has been received from controller, returning default = joint_max_speed` --> `Ignoring any future reports of error` : race condition between GUI and controller startup. Settles on its own.
- Split package warning (`com.ur.urcap.api.domain.value; version=1.9.0`) : OSGi packaging quirk in URSim itself. Not your problem.

## Known issues & quirks

- **Fullscreen tearing** : drag the window corner to resize instead of clicking maximize. If unusable, log out and switch to Ubuntu on Xorg at the login screen.
- **"No controller" in Polyscope** : URControl is crashing in the background. Check `URControl.log`. Most common cause: the `/usr/local/urcontrol/dynlibs` symlink is missing or points to a moved directory.
- **System `apt upgrade` may break things** : the patched `libxmlrpc-c-ur` package and the modern `lib32gcc-s1` are held together by a stripped dependency constraint. If an upgrade reinstalls either, re-run step 3 and `sudo apt --fix-broken install`.
- **Don't move the URSim directory** after install : the dynlibs symlink is absolute. If you must move it, update the symlink target (`sudo ln -sfn "$URSIM/dynlibs" /usr/local/urcontrol/dynlibs`).

## Verifying it works

After launching:

1. Polyscope GUI loads to 100% and shows the robot model (not "No Controller").
2. In Polyscope, the robot mode transitions from `NO_CONTROLLER` --> `POWER_ON` --> `RUNNING` (visible in logs as `TRACE_ROUTE RobotMode[...robotMode=RUNNING...]`).
3. From a separate terminal, this should print a 6-element pose:
   ```bash
   python -c "from rtde_receive import RTDEReceiveInterface; print(RTDEReceiveInterface('127.0.0.1').getActualTCPPose())"
   ```

---

*Document last verified: June 2026. URSim and Linux package versions drift over time : if a step fails on a later release, check the version stanza at the top of this document against your current setup.*