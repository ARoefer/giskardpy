# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions

If you are on Ubuntu 14.04(Trusty) add the LLVM repositories to your package list by running
```
sudo add-apt-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main"
sudo add-apt-repository "deb-src http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main"
```

Install symengine + symengine.py by running 
```
wget https://raw.githubusercontent.com/ARoefer/giskardpy/gebsyas/install_symengine.sh
install_symengine.sh
```
in a location of your choosing. The script will install LLVM 6.0, clone the symengine repositories to your selected location and automatically build them. 

*A reboot may be necessary for Python to find the symengine libraries.*

Install pybullet:
```
sudo pip install pybullet
```

Now create the workspace
```
source /opt/ros/kinetic/setup.bash          # start using ROS kinetic
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace
cd src                                      # go to source directory of workspace
wstool init                                 # init rosinstall
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/master/rosinstall/catkin.rosinstall
                                            # update rosinstall file
wstool update                               # pull source repositories
rosdep install --ignore-src --from-paths .  # install dependencies available through apt
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

### Tests
Run
```
catkin build --catkin-make-args run_tests  # build packages
```
