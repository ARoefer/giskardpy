import rospy

from giskardpy.plugin import PluginContainer
from giskardpy.plugin_action_server import ActionServer
from giskardpy.application import ROSApplication
from giskardpy.plugin_instantaneous_controller import JointControllerPlugin, CartesianControllerPlugin, \
    CartesianBulletControllerPlugin
from giskardpy.plugin_fk import FKPlugin
from giskardpy.plugin_interactive_marker import InteractiveMarkerPlugin
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.plugin_pybullet import PyBulletPlugin
from giskardpy.process_manager import ProcessManager

if __name__ == '__main__':
    rospy.init_node('muh')

    # roots = ['base_footprint']
    # tips = ['gripper_tool_frame']
    roots = ['base_link', 'base_link']
    tips = ['r_gripper_tool_frame', 'l_gripper_tool_frame']
    fk_identifier = 'fk'
    cartesian_goal_identifier = 'cartesian_goal'

    js_identifier = 'js'
    joint_goal_identifier = 'joint_goal'
    trajectory_identifier = 'traj'
    time_identifier = 'time'
    next_cmd_identifier = 'motor'
    collision_identifier = 'collision'
    closest_point_identifier = 'cpi'

    pm = ProcessManager()
    pm.register_plugin('js', JointStatePlugin(js_identifier=js_identifier, time_identifier=time_identifier,
                                              next_cmd_identifier=next_cmd_identifier))
    pm.register_plugin('action server', ActionServer(js_identifier=js_identifier,
                                                     trajectory_identifier=trajectory_identifier,
                                                     cartesian_goal_identifier=cartesian_goal_identifier,
                                                     time_identifier=time_identifier))
    pm.register_plugin('bullet', PluginContainer(PyBulletPlugin(js_identifier=js_identifier, collision_identifier=collision_identifier,
                                                closest_point_identifier=closest_point_identifier)))
    pm.register_plugin('fk', FKPlugin(roots, tips, js_identifier=js_identifier, fk_identifier=fk_identifier))
    # pm.register_plugin('joint controller', JointControllerPlugin(js_identifier=js_identifier,
    #                                                              goal_identifier=joint_goal_identifier,
    #                                                              next_cmd_identifier=next_cmd_identifier))
    # pm.register_plugin('cartesian controller', CartesianControllerPlugin(roots, tips,
    #                                                                      fk_identifier=fk_identifier,
    #                                                                      goal_identifier=cartesian_goal_identifier,
    #                                                                      js_identifier=js_identifier,
    #                                                                      next_cmd_identifier=next_cmd_identifier))
    pm.register_plugin('cart bullet controller', CartesianBulletControllerPlugin(roots, tips,
                                                                                 fk_identifier=fk_identifier,
                                                                                 goal_identifier=cartesian_goal_identifier,
                                                                                 js_identifier=js_identifier,
                                                                                 next_cmd_identifier=next_cmd_identifier,
                                                                                 collision_identifier=collision_identifier,
                                                                                 closest_point_identifier=closest_point_identifier))
    pm.register_plugin('interactive marker', InteractiveMarkerPlugin(roots, tips))

    app = ROSApplication(pm)
    app.run()