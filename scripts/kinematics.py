#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np

class xArm7_kinematics():
    def __init__(self):

        self.l1 = 0.267
        self.l2 = 0.293
        self.l3 = 0.0525
        self.l4 = 0.3512
        self.l5 = 0.1232

        self.f1 = 0.2225 #(rad) (=12.75deg)
        self.f2 = 0.6646 #(rad) (=38.08deg)

        self.l4costheta1= 0.3425
        self.l4sintheta1= 0.0775
        self.l5costheta2= 0.0969
        self.l5sintheta2= 0.0759

        pass

    def compute_jacobian(self, r_joints_array):

        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4
        l5 = self.l5

        f1=self.f1
        f2=self.f2

        """ DEFINE THE ANGLE JOINTS """
        q1 = r_joints_array[0]
        q2 = r_joints_array[1]
        q3 = r_joints_array[2]
        q4 = r_joints_array[3]
        q5 = r_joints_array[4]
        q6 = r_joints_array[5]
        q7 = r_joints_array[6]

        """ CREATE THE SINEs & COSINEs """
        c1 = np.cos(q1)
        c2 = np.cos(q2)
        c3 = np.cos(q3)
        c4 = np.cos(q4)
        c5 = np.cos(q5)
        c6 = np.cos(q6)
        c7 = np.cos(q7)
        cf1=np.cos(f1)
        cf2=np.cos(f2)
        cf2q6=np.cos(f2-q6)

        s1 = np.sin(q1)
        s2 = np.sin(q2)
        s3 = np.sin(q3)
        s4 = np.sin(q4)
        s5 = np.sin(q5)
        s6 = np.sin(q6)
        s7 = np.sin(q7)
        sf1= np.sin(f1)
        sf2= np.sin(f2)
        sf2q6=np.sin(f2-q6)


        J_11 = l5*sf2q6*s5*(c1*c3 - c2*s1*s3) - (l4*cf1 + l5*cf2q6)*(c1*s3*s4 - c4*s1*s2 + c2*c3*s1*s4) - l3*c1*s3 - l2*s1*s2 - (s1*s2*s4 + c1*c4*s3 + c2*c3*c4*s1)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - l3*c2*c3*s1
        J_12 = c1*(c2*s4 - c3*c4*s2)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - c1*(c2*c4 + c3*s2*s4)*(l4*cf1 + l5*cf2q6) + l2*c1*c2 - l3*c1*c3*s2 - l5*sf2q6*c1*s2*s3*s5
        J_13 = - c4*(c3*s1 + c1*c2*s3)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - s4*(c3*s1 + c1*c2*s3)*(l4*cf1 + l5*cf2q6) - l3*c3*s1 - l5*sf2q6*s5*(s1*s3 - c1*c2*c3) - l3*c1*c2*s3
        J_14 = (s1*s3*s4 + c1*c4*s2 - c1*c2*c3*s4)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) + (l4*cf1 + l5*cf2q6)*(c1*s2*s4 - c4*s1*s3 + c1*c2*c3*c4)
        J_15 = l5*sf2q6*c5*(c3*s1 + c1*c2*s3) - l5*sf2q6*s5*(c1*s2*s4 - c4*s1*s3 + c1*c2*c3*c4)
        J_16 = - l5*sf2q6*(s1*s3*s4 + c1*c4*s2 - c1*c2*c3*s4) - l5*s5*cf2q6*(c3*s1 + c1*c2*s3) - l5*c5*cf2q6*(c1*s2*s4 - c4*s1*s3 + c1*c2*c3*c4)
        J_17 = 0

        J_21 = (c1*s2*s4 - c4*s1*s3 + c1*c2*c3*c4)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - (l4*cf1 + l5*cf2q6)*(s1*s3*s4 + c1*c4*s2 - c1*c2*c3*s4) + l2*c1*s2 - l3*s1*s3 + l5*sf2q6*s5*(c3*s1 + c1*c2*s3) + l3*c1*c2*c3
        J_22 = s1*(c2*s4 - c3*c4*s2)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - s1*(c2*c4 + c3*s2*s4)*(l4*cf1 + l5*cf2q6) + l2*c2*s1 - l3*c3*s1*s2 - l5*sf2q6*s1*s2*s3*s5
        J_23 = c4*(c1*c3 - c2*s1*s3)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) + s4*(c1*c3 - c2*s1*s3)*(l4*cf1 + l5*cf2q6) + l3*c1*c3 + l5*sf2q6*s5*(c1*s3 + c2*c3*s1) - l3*c2*s1*s3
        J_24 = (l4*cf1 + l5*cf2q6)*(s1*s2*s4 + c1*c4*s3 + c2*c3*c4*s1) - (c1*s3*s4 - c4*s1*s2 + c2*c3*s1*s4)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2)
        J_25 = - l5*sf2q6*c5*(c1*c3 - c2*s1*s3) - l5*sf2q6*s5*(s1*s2*s4 + s1*c4*s3 + c2*c3*c4*s1)
        J_26 = l5*sf2q6*(c1*s3*s4 - c4*s1*s2 + c2*c3*s1*s4) + l5*s5*cf2q6*(c1*c3 - c2*s1*s3) - l5*c5*cf2q6*(s1*s2*s4 + c1*c4*s3 + c2*c3*c4*s1)
        J_27 = 0

        J_31 = 0
        J_32 = (c4*s2 - c2*c3*s4)*(l4*cf1 + l5*cf2q6) - (s2*s4 + c2*c3*c4)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) - l2*s2 - l3*c2*c3 - l5*sf2q6*c2*s3*s5
        J_33 = l3*s2*s3 + c4*s2*s3*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) + s2*s3*s4*(l4*cf1 + l5*cf2q6) - l5*sf2q6*c3*s2*s5
        J_34 = (c2*c4 + c3*s2*s4)*(l4*sf1 - l5*cf2*c5*s6 + l5*c5*c6*sf2) + (c2*s4 - c3*c4*s2)*(l4*cf1 + l5*cf2q6)
        J_35 = - l5*sf2q6*s5*(c2*s4 - c3*c4*s2) - l5*sf2q6*c5*s2*s3
        J_36 = l5*s2*s3*s5*cf2q6 - l5*c5*cf2q6*(c2*s4 - c3*c4*s2) - l5*sf2q6*(c2*c4 + c3*s2*s4)
        J_37 = 0

        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ]])
        return J

    def tf_A01(self, r_joints_array):
        l1= self.l1

        q1=r_joints_array[0]

        c1=np.cos(q1)
        s1=np.sin(q1)

        tf = np.matrix([[c1 , -s1 , 0 , 0],\
                        [s1 , c1 , 0 , 0],\
                        [0 , 0 , 1 , l1],\
                        [0 , 0 , 0 , 1]])
        return tf

    def tf_A02(self, r_joints_array):
        l1= self.l1

        q1=r_joints_array[0]
        q2=r_joints_array[1]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)

        tf_A12 = np.matrix([[c2 , -s2 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [-s2 , -c2 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        l1= self.l1
        l2= self.l2

        q1=r_joints_array[0]
        q2=r_joints_array[1]
        q3=r_joints_array[2]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)
        c3=np.cos(q3)
        s3=np.sin(q3)


        tf_A23 = np.matrix([[c3 , -s3 , 0 , 0],\
                            [0 , 0 , -1 , -l2],\
                            [s3 , c3 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        l1= self.l1
        l2= self.l2
        l3= self.l3

        q1=r_joints_array[0]
        q2=r_joints_array[1]
        q3=r_joints_array[2]
        q4=r_joints_array[3]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)
        c3=np.cos(q3)
        s3=np.sin(q3)
        c4=np.cos(q4)
        s4=np.sin(q4)


        tf_A34 = np.matrix([[c4 , -s4 , 0 , l3],\
                            [0 , 0 , -1 , 0],\
                            [s4 , c4 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        l1= self.l1
        l2= self.l2
        l3= self.l3
        l4= self.l4

        l4sintheta1=self.l4sintheta1
        l4costheta1=self.l4costheta1

        q1=r_joints_array[0]
        q2=r_joints_array[1]
        q3=r_joints_array[2]
        q4=r_joints_array[3]
        q5=r_joints_array[4]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)
        c3=np.cos(q3)
        s3=np.sin(q3)
        c4=np.cos(q4)
        s4=np.sin(q4)
        c5=np.cos(q5)
        s5=np.sin(q5)

        tf_A45 = np.matrix([[c5 , -s5 , 0 , l4sintheta1],\
                            [0 , 0 , -1 , -l4costheta1],\
                            [s5 , c5 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        l1= self.l1
        l2= self.l2
        l3= self.l3
        l4= self.l4

        q1=r_joints_array[0]
        q2=r_joints_array[1]
        q3=r_joints_array[2]
        q4=r_joints_array[3]
        q5=r_joints_array[4]
        q6=r_joints_array[5]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)
        c3=np.cos(q3)
        s3=np.sin(q3)
        c4=np.cos(q4)
        s4=np.sin(q4)
        c5=np.cos(q5)
        s5=np.sin(q5)
        c6=np.cos(q6)
        s6=np.sin(q6)

        tf_A56 = np.matrix([[c6 , -s6 , 0 , 0],\
                            [0 , 0 , -1 , 0],\
                            [s6 , c6 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        l1= self.l1
        l2= self.l2
        l3= self.l3
        l4= self.l4
        l5= self.l5

        l5sintheta2=self.l5sintheta2
        l5costheta2=self.l5costheta2

        q1=r_joints_array[0]
        q2=r_joints_array[1]
        q3=r_joints_array[2]
        q4=r_joints_array[3]
        q5=r_joints_array[4]
        q6=r_joints_array[5]
        q7=r_joints_array[6]

        c1=np.cos(q1)
        s1=np.sin(q1)
        c2=np.cos(q2)
        s2=np.sin(q2)
        c3=np.cos(q3)
        s3=np.sin(q3)
        c4=np.cos(q4)
        s4=np.sin(q4)
        c5=np.cos(q5)
        s5=np.sin(q5)
        c6=np.cos(q6)
        s6=np.sin(q6)
        c7=np.cos(q7)
        s7=np.sin(q7)

        tf_A67 = np.matrix([[c7 , -s7 , 0 , l5sintheta2],\
                            [0 , 0 , 1 , l5costheta2],\
                            [-s7 , -c7 , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A06(r_joints_array), tf_A67 )
        return tf


    def get_position_link3(self, q):
        A03 = self.tf_A03(q)
        pos = A03[0:3, 3]
        return np.array([pos[0, 0], pos[1, 0], pos[2, 0]])

    def get_position_link4(self, q):
        A04 = self.tf_A04(q)
        pos = A04[0:3, 3]
        return np.array([pos[0, 0], pos[1, 0], pos[2, 0]])

    def get_position_link5(self, q):
        A05 = self.tf_A05(q)
        pos = A05[0:3, 3]
        return np.array([pos[0, 0], pos[1, 0], pos[2, 0]])

    def compute_jacobian_link3(self, q):
        J_full = self.compute_jacobian(q)
        J3 = np.copy(J_full)
        # Μηδενίζουμε τη συμβολή των αρθρώσεων 4,5,6,7
        J3[:, 3:] = 0
        return J3

    def compute_jacobian_link4(self, q):
        J_full = self.compute_jacobian(q)
        J4 = np.copy(J_full)
        # Μηδενίζουμε τη συμβολή των αρθρώσεων 5,6,7
        J4[:, 4:] = 0
        return J4

    def compute_jacobian_link5(self, q):
        J_full = self.compute_jacobian(q)
        J5 = np.copy(J_full)
        # Μηδενίζουμε τη συμβολή των αρθρώσεων 6,7
        J5[:, 5:] = 0
        return J5
