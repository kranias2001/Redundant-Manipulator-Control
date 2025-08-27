#!/usr/bin/env python3

"""
Start ROS node to control xArm7 for periodic y-axis movement with obstacle avoidance.
Publishes end-effector position and velocity for real-time monitoring in rqt_plot.
"""

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
import numpy as np
from numpy.linalg import pinv
from kinematics import xArm7_kinematics
from gazebo_msgs.msg import LinkStates


class xArm7_controller():
    def __init__(self, rate):
        self.kinematics = xArm7_kinematics()

        self.joint_angpos = [0, 0, 0, 0, 0, 0, 0]
        self.joint_angvel = [0, 0, 0, 0, 0, 0, 0]
        self.joint_states = JointState()
        self.model_states = ModelStates()
        self.y_obstacle1=0.0
        self.y_obstacle2=0.0

        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)

        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)

        self.ee_pos_x_pub = rospy.Publisher('/ee_position/x', Float64, queue_size=1)
        self.ee_pos_y_pub = rospy.Publisher('/ee_position/y', Float64, queue_size=1)
        self.ee_pos_z_pub = rospy.Publisher('/ee_position/z', Float64, queue_size=1)

        self.ee_vel_x_pub = rospy.Publisher('/ee_velocity/vx', Float64, queue_size=1)
        self.ee_vel_y_pub = rospy.Publisher('/ee_velocity/vy', Float64, queue_size=1)
        self.ee_vel_z_pub = rospy.Publisher('/ee_velocity/vz', Float64, queue_size=1)

        self.ee_pose_x_real_pub = rospy.Publisher('/ee_real_position/x', Float64, queue_size=1)
        self.ee_pose_y_real_pub = rospy.Publisher('/ee_real_position/y', Float64, queue_size=1)
        self.ee_pose_z_real_pub = rospy.Publisher('/ee_real_position/z', Float64, queue_size=1)

        self.ee_vel_x_real_pub = rospy.Publisher('/ee_real_velocity/vx', Float64, queue_size=1)
        self.ee_vel_y_real_pub = rospy.Publisher('/ee_real_velocity/vy', Float64, queue_size=1)
        self.ee_vel_z_real_pub = rospy.Publisher('/ee_real_velocity/vz', Float64, queue_size=1)

        self.ee_vel_x_jacob_pub = rospy.Publisher('/ee_velocity_jacobian/vx', Float64, queue_size=1)
        self.ee_vel_y_jacob_pub = rospy.Publisher('/ee_velocity_jacobian/vy', Float64, queue_size=1)
        self.ee_vel_z_jacob_pub = rospy.Publisher('/ee_velocity_jacobian/vz', Float64, queue_size=1)

        self.link_states_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_callback)

        # Distance from obstacles
        self.dist_j3_obs1_pub = rospy.Publisher('/joint3/obstacle1_distance', Float64, queue_size=1)
        self.dist_j3_obs2_pub = rospy.Publisher('/joint3/obstacle2_distance', Float64, queue_size=1)
        self.dist_j4_obs1_pub = rospy.Publisher('/joint4/obstacle1_distance', Float64, queue_size=1)
        self.dist_j4_obs2_pub = rospy.Publisher('/joint4/obstacle2_distance', Float64, queue_size=1)
        self.dist_j5_obs1_pub = rospy.Publisher('/joint5/obstacle1_distance', Float64, queue_size=1)
        self.dist_j5_obs2_pub = rospy.Publisher('/joint5/obstacle2_distance', Float64, queue_size=1)


        self.period = 1.0 / rate
        self.pub_rate = rospy.Rate(rate)

        self.x_fixed = 0.617
        self.z_fixed = 0.199
        self.T_total = 6.0
        self.phases = [
            (0.17, -0.226),
            (-0.226, 0.17)
        ]
        self.phase_time = self.T_total / len(self.phases)
        self.start_time = rospy.get_time()
        self.current_phase = 0

        self.publish()

    def link_states_callback(self, msg):
        try:
            index = msg.name.index("xarm7::link7")  # Ή "xarm7::link7" ανάλογα με το namespace σου
            pos = msg.pose[index].position
            vel = msg.twist[index].linear

            self.ee_pose_x_real_pub.publish(pos.x)
            self.ee_pose_y_real_pub.publish(pos.y)
            self.ee_pose_z_real_pub.publish(pos.z)

            self.ee_vel_x_real_pub.publish(vel.x)
            self.ee_vel_y_real_pub.publish(vel.y)
            self.ee_vel_z_real_pub.publish(vel.z)

        except ValueError:
            rospy.logwarn("link7 not found in /gazebo/link_states")


    def quintic_polynomial(self, t, T, y0, yf):
        y0_dot = 0
        yf_dot = 0
        y0_ddot = 0
        yf_ddot = 0

        a0 = y0
        a1 = y0_dot
        a2 = y0_ddot / 2
        a3 = (20*(yf - y0) - (8*yf_dot + 12*y0_dot)*T - (3*y0_ddot - yf_ddot)*T**2) / (2*T**3)
        a4 = (30*(y0 - yf) + (14*yf_dot + 16*y0_dot)*T + (3*y0_ddot - 2*yf_ddot)*T**2) / (2*T**4)
        a5 = (12*(yf - y0) - (6*yf_dot + 6*y0_dot)*T - (y0_ddot - yf_ddot)*T**2) / (2*T**5)

        pos = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        vel = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        acc = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        return pos, vel, acc

    def joint_states_callback(self, msg):
        self.joint_states = msg
        self.joint_angpos = list(msg.position)

    def model_states_callback(self, msg):
        self.model_states = msg
        self.y_obstacle1 = msg.pose[1].position.y
        self.y_obstacle2 = msg.pose[2].position.y

    def publish(self):
        j2 = 0.7 ; j4 = np.pi/2
        j6 = -(j2-j4)
        self.joint_angpos = [0, j2, 0, j4, 0, j6, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()
        start_time_absolute = rostime_now.secs + rostime_now.nsecs / 1e9

        self.mode = 'INIT'
        self.start_time = rospy.get_time()
        self.T_init = 3.0

        while not rospy.is_shutdown():
            rostime_now = rospy.get_rostime()
            now = rostime_now.secs + rostime_now.nsecs / 1e9
            t_now = now - start_time_absolute

            self.obstacleMed= ( self.y_obstacle1 + 1.2* self.y_obstacle2)/2
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            pinvJ = pinv(J)

            if self.mode == 'INIT':
                t_elapsed = rospy.get_time() - self.start_time
                if t_elapsed <= self.T_init:
                    y0, yf = 0.0, 0.168
                    y_pos_desired, y_vel_desired, _ = self.quintic_polynomial(t_elapsed, self.T_init, y0, yf)
                    self.y_vel_desired = y_vel_desired
                else:
                    print("Finished INIT phase, starting PERIODIC motion...")
                    self.mode = 'PERIODIC'
                    self.start_time = rospy.get_time()
                    self.current_phase = 0
                    continue

            elif self.mode == 'PERIODIC':
                now_phase = rospy.get_time() - self.start_time
                if self.current_phase < len(self.phases):
                    phase_start = self.current_phase * self.phase_time
                    phase_end = (self.current_phase + 1) * self.phase_time

                    if phase_start <= now_phase < phase_end:
                        t_phase = now_phase - phase_start
                        y0, yf = self.phases[self.current_phase]
                        y_pos_desired , y_vel_desired, _ = self.quintic_polynomial(t_phase, self.phase_time, y0, yf)
                        self.y_vel_desired = y_vel_desired

                    else:
                        self.current_phase += 1
                        if self.current_phase >= len(self.phases):
                            self.current_phase = 0
                            self.start_time = rospy.get_time()
                        continue

            x_target = self.x_fixed
            y_target = self.obstacleMed + y_pos_desired
            z_target = self.z_fixed

            p_0E = self.kinematics.tf_A07(self.joint_angpos)[0:3, 3]
            current_x = p_0E[0, 0]
            current_y = p_0E[1, 0]
            current_z = p_0E[2, 0]

            Kpx, Kpz = 0.8, 0.8
            error_x = x_target - current_x
            error_z = z_target - current_z

            p1_dot_desired = np.transpose(np.matrix([Kpx*error_x, y_vel_desired, Kpz*error_z]))

            task1 = np.dot(pinvJ, p1_dot_desired)

                        # [1] Κινηματική και Ιακωβιανή
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            pinvJ = pinv(J)

            # [2] Θέσεις των αρθρώσεων από forward kinematics
            P0_3 = self.kinematics.get_position_link3(self.joint_angpos)
            P0_4 = self.kinematics.get_position_link4(self.joint_angpos)
            P0_5 = self.kinematics.get_position_link5(self.joint_angpos)

            # Απόσταση από Obstacle 1 (y = -0.2)
            dist_j3_obs1 = abs(P0_3[1] - (-0.2))
            dist_j4_obs1 = abs(P0_4[1] - (-0.2))
            dist_j5_obs1 = abs(P0_5[1] - (-0.2))

            # Απόσταση από Obstacle 2 (y = +0.2)
            dist_j3_obs2 = abs(P0_3[1] - (+0.2))
            dist_j4_obs2 = abs(P0_4[1] - (+0.2))
            dist_j5_obs2 = abs(P0_5[1] - (+0.2))

            # Δημοσίευση σε topics για plotting
            self.dist_j3_obs1_pub.publish(dist_j3_obs1)
            self.dist_j3_obs2_pub.publish(dist_j3_obs2)
            self.dist_j4_obs1_pub.publish(dist_j4_obs1)
            self.dist_j4_obs2_pub.publish(dist_j4_obs2)
            self.dist_j5_obs1_pub.publish(dist_j5_obs1)
            self.dist_j5_obs2_pub.publish(dist_j5_obs2)

            # [3] Παράμετροι μήκους xArm7
            l1 = 0.267
            l2 = 0.293
            l3 = 0.0525
            l4 = 0.3512
            l5 = 0.1232

            # [4] Προ-υπολογισμός τριγωνομετρικών
            q = self.joint_angpos
            c1, s1 = np.cos(q[0]), np.sin(q[0])
            c2, s2 = np.cos(q[1]), np.sin(q[1])
            c3, s3 = np.cos(q[2]), np.sin(q[2])
            c4, s4 = np.cos(q[3]), np.sin(q[3])
            sin1 = np.sin(0.2225)
            cos1 = np.cos(0.2225)

            # [5] Υπολογισμός obstacleMed
            obstacle1 = [0, -0.200000, 0]
            obstacle2 = [0, 0.200000, 0]
            obstacleMed = (obstacle1[1] + 1.2 * obstacle2[1]) / 2

            # [6] Κέρδη
            kc1, kc2, kc3 = [30.0, 800.0, 0.0]

            # [7] Συνάρτηση κόστους (για debug/logging αν θέλεις)
            V = kc1*(P0_3[1] - obstacleMed)**2 + \
                kc2*(P0_4[1] - obstacleMed)**2 + \
                kc3*(P0_5[1] - obstacleMed)**2

            # [8] Gradient vectors
            v1 = 2*kc1*(P0_3[1]-obstacleMed)*l2*c1*s2 + \
                 2*kc2*(P0_4[1]-obstacleMed)*(l3*c1*c2*c2 - l3*s1*s3 + l2*c1*s2) + \
                 2*kc3*(P0_5[1]-obstacleMed)*(l4*sin1*(c1*c2*c3*c4 - s1*s3*c4 + c1*s2*s4)
                                                  - l4*cos1*(-c1*c2*c3*s4 + s1*s3*s4 + c1*s2*c4)
                                                  + l3*(c1*c2*c3 - s1*s3) + l2*c1*s2)

            v2 = 2*kc1*(P0_3[1]-obstacleMed)*l2*s1*c2 + \
                 2*kc2*(P0_4[1]-obstacleMed)*(-l3*s1*s2*c3 + l2*s1*c2) + \
                 2*kc3*(P0_5[1]-obstacleMed)*(l4*sin1*(-s1*s2*c3*c4 + s1*c2*s4)
                                                  - l4*cos1*(s1*s2*c3*s4 + s1*c2*c4)
                                                  + l3*(-s1*s2*c3) + l2*s1*c2)

            v3 = 2*kc2*(P0_4[1]-obstacleMed)*(-l3*s1*c2*s3 + l3*c1*c3) + \
                 2*kc3*(P0_5[1]-obstacleMed)*(l4*sin1*(-s1*c2*s3*c4 + c1*c3*c4)
                                                  - l4*cos1*(s1*c2*s3*s4 - c1*c3*s4)
                                                  + l3*(-s1*c2*s3 + c1*c3))

            v4 = 2*kc3*(P0_5[1]-obstacleMed)*(l4*sin1*(-s1*c2*c3*s4 + s1*s2*c4 - c1*s3*s4)
                                                  - l4*cos1*(-s1*c2*c3*c4 - c1*s3*c4 - s1*s2*s4))

            # [9] Τελικά gradient vector και q_dot
            gradient_vector =  -1 * np.transpose(np.matrix([
                v1, v2, v3, v4, 0.0, 0.0, 0.0
            ]))

            I7 = np.identity(7)
            task2 = (I7 - pinvJ @ J) @ gradient_vector
            ang_vel_desired = task1 + task2

            # [10] Ενημέρωση ταχυτήτων αρθρώσεων
            for i in range(7):
                self.joint_angvel[i] = ang_vel_desired[i, 0]

            v_0E = J[0:3, :] @ np.reshape(self.joint_angvel, (7, 1))

            self.ee_vel_x_jacob_pub.publish(v_0E[0, 0])
            self.ee_vel_y_jacob_pub.publish(v_0E[1, 0])
            self.ee_vel_z_jacob_pub.publish(v_0E[2, 0])
            self.ee_pos_x_pub.publish(current_x)
            self.ee_pos_y_pub.publish(current_y)
            self.ee_pos_z_pub.publish(current_z)

            self.ee_vel_x_pub.publish(v_0E[0, 0])
            self.ee_vel_y_pub.publish(v_0E[1, 0])
            self.ee_vel_z_pub.publish(v_0E[2, 0])

            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev) / 1e9

            self.joint_angpos = np.add(self.joint_angpos, [index * dt for index in self.joint_angvel])

            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])


            self.pub_rate.sleep()

    def turn_off(self):
        pass

def controller_py():
    rospy.init_node('controller_node', anonymous=True)
    rate = rospy.get_param("/rate", 1000)
    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
