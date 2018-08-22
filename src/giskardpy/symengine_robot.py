from collections import namedtuple, OrderedDict
import numpy as np
import symengine_wrappers as spw
from urdf_parser_py.urdf import URDF, Box, Sphere, Mesh

from giskardpy.input_system import JointStatesInput
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'type', 'frame'])
Gripper = namedtuple('Gripper', ['name', 'pose', 'opening', 'height', 'max_opening', 'link_name', 'reach', 'pivot_position'])
Camera = namedtuple('Camera', ['name', 'pose', 'hfov', 'near', 'far'])

def hacky_urdf_parser_fix(urdf_str):
    # TODO this function is inefficient but the tested urdf's aren't big enough for it to be a problem
    fixed_urdf = ''
    delete = False
    black_list = ['transmission']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    for line in urdf_str.split('\n'):
        if len([x for x in black_open if x in line]) > 0:
            delete = True
        if len([x for x in black_close if x in line]) > 0:
            delete = False
            continue
        if not delete:
            fixed_urdf += line + '\n'
    return fixed_urdf


class Robot(object):
    def __init__(self, urdf, default_joint_vel_limit=0):
        self.default_joint_vel_limit = default_joint_vel_limit
        self.default_weight = 0.0001
        self.fks = {}
        if urdf.endswith('.urdf'):
            self._load_from_urdf_file(urdf)
        else:
            self._load_from_urdf_string(urdf)

    def get_name(self):
        return self._urdf_robot.name

    def _load_from_urdf_string(self, urdf_str):
        return self._load_from_urdf(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_str)))

    def _load_from_urdf_file(self, urdf_file):
        with open(urdf_file, 'r') as f:
            urdf_string = f.read()
        return self._load_from_urdf_string(urdf_string)

    def _load_from_urdf(self, urdf_robot):
        self._urdf_robot = urdf_robot
        self._create_sym_frames()
        self._create_constraints()

    def set_joint_symbol_map(self, joint_states_input=None):
        if joint_states_input is not None:
            self.joint_states_input = joint_states_input
            for joint_name, joint in self._joints.items():
                # for joint_name, joint_symbol in joint_states_input.joint_map.items():
                new_symbol = None
                if joint.symbol is not None and joint_name in self.joint_states_input.joint_map:
                    new_symbol = self.joint_states_input.joint_map[joint_name]
                self._joints[joint_name] = Joint(new_symbol,
                                                 joint.velocity_limit,
                                                 joint.lower,
                                                 joint.upper,
                                                 joint.type,
                                                 # TODO this line is relatively slow
                                                 joint.frame.subs(self.joint_states_input.joint_map))
            self._create_constraints()

    def _create_sym_frames(self):
        self._joints = {}
        joint_map = {}
        for joint_name, joint in self._urdf_robot.joint_map.items():
            # TODO use a dict here?
            joint_symbol = None
            if joint.type in ['revolute', 'continuous', 'prismatic']:
                if joint.mimic is None:
                    joint_map[joint_name] = spw.Symbol(joint_name)
                    joint_symbol = joint_map[joint_name]
                else:
                    joint_map[joint.mimic.joint] = spw.Symbol(joint.mimic.joint)
                    multiplier = 1 if joint.mimic.multiplier is None else joint.mimic.multiplier
                    offset = 0 if joint.mimic.offset is None else joint.mimic.offset
                    mimic = joint_map[joint.mimic.joint] * multiplier + offset

            if joint.type in ['fixed', 'revolute', 'continuous', 'prismatic']:
                if joint.origin is not None:
                    xyz = joint.origin.xyz if joint.origin.xyz is not None else [0,0,0]
                    rpy = joint.origin.rpy if joint.origin.rpy is not None else [0,0,0]
                    joint_frame = spw.translation3(*xyz) * spw.rotation3_rpy(*rpy)
                else:
                    joint_frame = spw.eye(4)
            else:
                raise Exception('Joint type "{}" is not supported by urdf parser.'.format(joint.type))

            if joint.type == 'revolute' or joint.type == 'continuous':
                if joint.mimic is None:
                    joint_frame *= spw.rotation3_axis_angle(spw.vector3(*joint.axis), joint_symbol)
                else:
                    joint_frame *= spw.rotation3_axis_angle(spw.vector3(*joint.axis), mimic)

            elif joint.type == 'prismatic':
                if joint.mimic is None:
                    joint_frame *= spw.translation3(*(spw.point3(*joint.axis) * joint_symbol)[:3])
                else:
                    joint_frame *= spw.translation3(*(spw.point3(*joint.axis) * mimic)[:3])

            # TODO simplify here?
            if joint.limit is not None:
                vel_limit = min(joint.limit.velocity, self.default_joint_vel_limit)
            else:
                vel_limit = None

            if joint.safety_controller is not None:
                lower_limit = joint.safety_controller.soft_lower_limit
                upper_limit = joint.safety_controller.soft_upper_limit
            else:
                if joint.limit is not None:
                    lower_limit = joint.limit.lower if joint.limit.lower is not None else None
                    upper_limit = joint.limit.upper if joint.limit.upper is not None else None
                else:
                    lower_limit = None
                    upper_limit = None
            self._joints[joint_name] = Joint(joint_symbol,
                                             vel_limit,
                                             lower_limit,
                                             upper_limit,
                                             joint.type,
                                             joint_frame)
        self.joint_states_input = JointStatesInput(joint_map)

    def _create_constraints(self):
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()
        for i, (joint_name, joint) in enumerate(self._joints.items()):
            if joint.symbol is not None:
                if joint.lower is not None and joint.upper is not None:
                    self.hard_constraints[joint_name] = HardConstraint(lower=joint.lower - joint.symbol,
                                                                       upper=joint.upper - joint.symbol,
                                                                       expression=joint.symbol)
                if joint.velocity_limit is None:
                    vel_limit = self.default_joint_vel_limit
                else:
                    vel_limit = min(joint.velocity_limit, self.default_joint_vel_limit)
                self.joint_constraints[joint_name] = JointConstraint(lower=-vel_limit,
                                                                     upper=vel_limit,
                                                                     weight=self.default_weight)

    def get_fk_expression(self, root_link, tip_link):
        if (root_link, tip_link) not in self.fks:
            jointsAndLinks = self._urdf_robot.get_chain(root_link, tip_link, True, True, True)
            fk = spw.eye(4)
            for i in range(1, len(jointsAndLinks), 2):
                joint_name = jointsAndLinks[i]
                fk *= self._joints[joint_name].frame
            self.fks[root_link, tip_link] = fk
        return self.fks[root_link, tip_link]

    def get_chain_joints(self, root, tip):
        return self._urdf_robot.get_chain(root, tip, True, False, False)

    def get_chain_links(self, root, tip):
        return self._urdf_robot.get_chain(root, tip, False, True, False)

    def get_chain_joint_symbols(self, root, tip):
        return [self.joint_states_input.joint_map[k] for k in self.get_chain_joints(root, tip)]

    def get_joint_symbol_map(self):
        return self.joint_states_input

    def get_joint_names(self):
        return [k for k, v in self._joints.items() if v.symbol is not None]

    def get_link_tree(self, root):
        links = set()
        joints = [root]
        for joint in joints:
            child_link = self._urdf_robot.joint_map[joint].child
            if child_link in self._urdf_robot.child_map:
                for j, l in self._urdf_robot.child_map[child_link]:
                    joints.append(j)
            link_collision = self._urdf_robot.link_map[child_link].collision
            if link_collision is not None:
                geo = link_collision.geometry
                if isinstance(geo, Box) and np.prod(geo.size) > 0.01 or \
                        isinstance(geo, Sphere) and geo.radius > 0.1 or \
                        isinstance(geo, Mesh):
                    links.add(child_link)

        return links

    def get_controllable_links(self, root_link):
        links = set()
        if root_link in self._urdf_robot.child_map:
            for j, l in self._urdf_robot.child_map[root_link]:
                joint_info = self._urdf_robot.joint_map[j]
                if joint_info.joint_type in ['revolute', 'continuous', 'prismatic']:
                    # links.add(l)
                    links.update(self.get_link_tree(j))
                links.update(self.get_controllable_links(l))
        return links


    def get_rnd_joint_state(self):
        js = {}

        def f(lower_limit, upper_limit):
            if lower_limit is None:
                return np.random.random() * np.pi * 2
            lower_limit = max(lower_limit, -10)
            upper_limit = min(upper_limit, 10)
            return (np.random.random() * (upper_limit - lower_limit)) + lower_limit

        for joint_name, joint in self._joints.items():
            if joint.symbol is not None:
                js[str(joint.symbol)] = f(joint.lower, joint.upper)
        return js

    def is_continuous(self, joint_name):
        return self._joints[joint_name].type == 'continuous'