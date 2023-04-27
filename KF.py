"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""

import numpy as np

class KF_4D(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
            dt: sampling time (time for 1 cycle)
            u_x: acceleration in x-direction
            u_y: acceleration in y-direction
            std_acc: process noise magnitude
            x_std_meas: standard deviation of the measurement in x-direction
            y_std_meas: standard deviation of the measurement in y-direction
        """
        self.dt = dt # sampling time
        self.u = np.matrix([[u_x],[u_y]]) # control input variables
        self.x = np.matrix([[0], [0], [0], [0], [0], [0], [0], [0]]) # intial State
        # [x, y, vx, vy, w, h, vw, vh].T
        
        # State Transition Matrix A (complete the definition)
        self.A = np.matrix([[1, 0, self.dt, 0, 0, 0, 0, 0],
                            [0, 1, 0, self.dt, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, self.dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, self.dt],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]])

        # Control Input Matrix B (defined for you)
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Measurement Mapping Matrix (complete the definition)
        # num_measurements x states
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance (defined for you)
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance (complete the definition)
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])

        #Initial Covariance Matrix (defined for you)
        self.P = np.eye(self.A.shape[1])


    def predict(self):
        ## complete this function
        # Update time state (self.x): x_k =Ax_(k-1) + Bu_(k-1) 
        self.x = self.A @ self.x + self.B @ self.u
        # Calculate error covariance (self.P): P= A*P*A' + Q 
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0:2]


    def update(self, z):
        ## complete this function
        # Calculate S = H*P*H'+R
        S = self.H @ self.P @ self.H.T + self.R
        # Calculate the Kalman Gain K = P * H'* inv(H*P*H'+R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update self.x
        self.x = self.x + K @ (z - self.H @ self.x)
        # Update error covariance matrix self.P
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x[0:2]
    

class KF_2D(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
            dt: sampling time (time for 1 cycle)
            u_x: acceleration in x-direction
            u_y: acceleration in y-direction
            std_acc: process noise magnitude
            x_std_meas: standard deviation of the measurement in x-direction
            y_std_meas: standard deviation of the measurement in y-direction
        """
        self.dt = dt # sampling time
        self.u = np.matrix([[u_x],[u_y]]) # control input variables
        self.x = np.matrix([[0], [0], [0], [0]]) # intial State
        
        # State Transition Matrix A (complete the definition)
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control Input Matrix B (defined for you)
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Measurement Mapping Matrix (complete the definition)
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance (defined for you)
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance (complete the definition)
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])

        #Initial Covariance Matrix (defined for you)
        self.P = np.eye(self.A.shape[1])


    def predict(self):
        ## complete this function
        # Update time state (self.x): x_k =Ax_(k-1) + Bu_(k-1) 
        self.x = self.A @ self.x + self.B @ self.u
        # Calculate error covariance (self.P): P= A*P*A' + Q 
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0:2]


    def update(self, z):
        ## complete this function
        # Calculate S = H*P*H'+R
        S = self.H @ self.P @ self.H.T + self.R
        # Calculate the Kalman Gain K = P * H'* inv(H*P*H'+R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update self.x
        self.x = self.x + K @ (z - self.H @ self.x)
        # Update error covariance matrix self.P
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x[0:2]