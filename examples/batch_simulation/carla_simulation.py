import carla
import math
import time
import numpy as np
import os
import cv2
import shutil
from filterpy.kalman import KalmanFilter
from scipy.stats import norm
import argparse
from abc import ABC, abstractmethod

unprotected_right_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000, # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 15,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_right_turn_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {'x': 4, 'y': -90, 'yaw': 0},
        'go_straight_ticks': 500, # * 10ms = 5s
        'turn_ticks': 250, # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {'x': 19, 'y': 28, 'yaw': 90},
        'go_straight_ticks': 400,  # * 10ms = 4s
        'turn_ticks': 200,        # * 10ms = 2s
        'after_turn_ticks': 350,  # * 10ms = 3.5s
        'throttle': {
            'straight': 0.52,
            'turn': 0.4,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {'x': 10, 'y': 35},
        'fov': '90'
    },
    'save_options': {
        'save_video': True,
        'save_images': True,
    }
}

unprotected_left_turn_config =  {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000, # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_left_turn_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {'x': 7.5, 'y': -90, 'yaw': 0},
        'go_straight_ticks': 500, # * 10ms = 5s
        'turn_ticks': 250, # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': -0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {'x': 3.5, 'y': 55, 'yaw': 180},
        'go_straight_ticks': 400,
        'turn_ticks': 200,
        'after_turn_ticks': 350,
        'throttle': {
            'straight': 0.53,
            'turn': 0.4,
            'after_turn': 0.52
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {'x': 10, 'y': 35},
        'fov': '90'
    },
    'save_options': {
        'save_video': True,
        'save_images': True,
    }
}


opposite_direction_merge_config =  {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000, # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/opposite_direction_merge_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {'x': 4, 'y': -90, 'yaw': 0},
        'go_straight_ticks': 500, # * 10ms = 5s
        'turn_ticks': 250, # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {'x': 8, 'y': 35, 'yaw': 180},
        'go_straight_ticks': 100,
        'turn_ticks': 400,
        'after_turn_ticks': 350,
        'throttle': {
            'straight': 0.52,
            'turn': 0.45,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': -0.4,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {'x': 10, 'y': 35},
        'fov': '90'
    },
    'save_options': {
        'save_video': True,
        'save_images': True,
    }
}


class BaseTracker(ABC):
    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        # Get vehicle dimensions from CARLA
        ego_bbox = ego_vehicle.bounding_box
        obs_bbox = obstacle_vehicle.bounding_box
        
        # Store vehicle dimensions
        self.ego_length = ego_bbox.extent.x * 2
        self.ego_width = ego_bbox.extent.y * 2
        self.obs_length = obs_bbox.extent.x * 2
        self.obs_width = obs_bbox.extent.y * 2
        
        self.dt = dt
        self.history = []
        self.history_ticks = []
        
    @abstractmethod
    def update(self, state, tick_number):
        pass
    
    @abstractmethod
    def predict_future_position(self, steps_ahead):
        pass
    
    def calculate_collision_probability(self, ego_state, obstacle_state):
        """
        Calculate collision probability between ego vehicle and obstacle
        ego_state: [x, y, theta in radians]
        obstacle_state: [x, y, theta in radians]
        """
        # %%
        import numpy as np
        # lo = 2
        # wo = 1
        # so = [-1.75, 2.0, 0.7853981633974483]

        # corner_dir = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
        # shape_matrix = np.array([[wo/2,0],[0,lo/2]]) 
        # # center_ori_matrix = np.array([ [np.cos(so[2]), np.sin(so[2]),so[0]],
        # #                       [np.sin(so[2]), np.cos(so[2]),so[1]],
        # #                       [0,0,1] ])
        # relative_rad = so[2]-np.pi/2
        # ori_matrix = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
        #                     [np.sin(relative_rad), np.cos(relative_rad)] ])
        # print(ori_matrix)
        # center_matrix = np.array([so[0],so[1]])

        # print(np.matmul(ori_matrix,shape_matrix))
        # np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) +np.tile(center_matrix,(4,1)).T
        # print(np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) )

        # %%
        from sympy import Point, Polygon

        # %%
        from scipy.spatial import distance
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        def Sort_List(l):
            # reverse = None (Sorts in Ascending order)
            # key is set to sort using second element of
            # sublist lambda has been used
            l.sort(key = lambda x: x[2])
            return l

        def collision_case(corner,w,l):
            x,y = corner
            d = 100
            close_point_list = [(x,l/2),(x,-l/2),(w/2,y),(w/2,-y)]
            for pts in close_point_list:
                dis = distance.euclidean(corner,pts)
                if dis<d:
                    d = dis
                    close_point = pts
            return (close_point,-d)

        def collision_dis_dir(corner,w,l):
        #     print('corner',corner)
            closet_point = (0,0)
            d = 100
            ego_corner = [(w/2,l/2),(w/2,-l/2),(-w/2,l/2),(-w/2,-l/2)]
            cor_x,cor_y = corner
            
            if np.abs(cor_x) < w/2 and np.abs(cor_y) < l/2 :
                closet_point,d = collision_case(corner,w,l)

            elif np.abs(cor_y) < l/2 and np.abs(cor_x) >= w/2 :
                d = -w/2-cor_x if cor_x<-w/2 else cor_x-w/2
                closet_point =  (-w/2,cor_y)  if cor_x<-w/2 else (w/2,cor_y)
            elif np.abs(cor_x) < w/2 and np.abs(cor_y) >= l/2 :
                d = -l/2-cor_y if cor_y<-l/2 else cor_y-l/2
                closet_point =  (cor_x,-l/2)  if cor_y<-l/2 else (cor_x,l/2)
            else:
                for ego_cor in ego_corner:
                    cor_dis = distance.euclidean(ego_cor, corner)
                    if cor_dis<d:
                        d = cor_dis
                        closet_point = ego_cor
        #     print(closet_point,d)
            return [closet_point,d]

        def object_tranformation(s1,s2):
            """
            kc's(orignal) frame {0}
            object s1's center as coordinate center {1}
            Transformation from {0} to {1} for object s2
            """ 
            relative_rad = np.pi/2 - s1[2]
            R = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
                            [np.sin(relative_rad), np.cos(relative_rad)]])
        #     print(R)
            # Obstacle coordinate transformation from {0} to {1}
            obs_center_homo = np.array([s2[0]-s1[0],s2[1]-s1[1]])
            obs_x,obs_y = np.matmul(R ,obs_center_homo)
            obs_theta = s2[2] + relative_rad
            so_f1 = [obs_x,obs_y,obs_theta]
            return so_f1


        def point_transformation(s1,p):
            """
            transfer point p back to the orginal coordinate from coordinate frame s1
            """

            relative_rad = np.pi/2 - s1[2]
            R = np.array([ [np.cos(-relative_rad), -np.sin(-relative_rad)],
                            [np.sin(-relative_rad), np.cos(-relative_rad)]])
            p = np.matmul(R,p)
            p= np.array([p[0]+s1[0],p[1]+s1[1]])
            return p

        def corners_cal(so_f1,lo,wo,corner_dir):
            obs_center_matrix = np.array([so_f1[0],so_f1[1]])
            shape_matrix = np.array([[wo/2,0],[0,lo/2]]) 

            relative_rad = so_f1[2]-np.pi/2
        #     print(relative_rad)
            ori_matrix = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
                                [np.sin(relative_rad), np.cos(relative_rad)] ])

            obs_corners = np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) +np.tile(obs_center_matrix,(4,1)).T #2*4
            return obs_corners

        def collision_point_rect(se, so, we = 1.5, le = 4, wo = 1.4, lo = 4,plot_flag = 0):
            """
            Input:
            - ego vehicle's state se = (x_e,y_e,theta_e)  at time k and shape (w_e,l_e)
            - obstacle's state mean so = (x_o,y_o,theta_o) at time k and shape prior (w_o,l_o)

            Output:
            - collision point and collision direction
            """
            # check theta is in radians and with in -pi to pi   
            if not isinstance(se[2], (int, float)) or not isinstance(so[2], (int, float)):
                raise ValueError("Theta values must be numeric.")
            # if se[2] < -np.pi*2 or se[2] > np.pi*2 or so[2] < -np.pi*2 or so[2] > np.pi*2:
            #     #raise ValueError(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")
            #     print(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")
            
            corner_dir = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

            # Transfer matrix from kc's frame to ego center
            so_f1 = object_tranformation(se,so)

            # 4 obstacle corner point to ego vehicle distance
            obs_corners = corners_cal(so_f1,lo,wo,corner_dir)
            obs_corners = obs_corners.T
            closest_point_dis = [ [tuple(corner)]+collision_dis_dir(corner,we,le) for corner in obs_corners]
            closest_point_dis = (Sort_List(closest_point_dis))
            # print('dis:',closest_point_dis[0][2])
            
            
            if plot_flag == 1:
                hull = ConvexHull(obs_corners)
                obs_corner_cov = obs_corners[hull.vertices]
                obs_corner_cov = np.append(obs_corner_cov,[obs_corner_cov[0]], axis=0)
                plt.plot(obs_corner_cov[:,0], obs_corner_cov[:,1], 'b--', lw=2)

                ego_corners = np.array([(we/2,le/2),(we/2,-le/2),(-we/2,le/2),(-we/2,-le/2)])
                hull = ConvexHull(ego_corners)
                ego_corners_cov = ego_corners[hull.vertices]
                ego_corners_cov = np.append(ego_corners_cov,[ego_corners_cov[0]], axis=0)
                plt.plot(ego_corners_cov[:,0], ego_corners_cov[:,1], 'r--', lw=2)
            
            
            # Transfer matrix from kc's frame to obstacle center frame
            se_f1 = object_tranformation(so,se)

            # 4 obstacle corner point to ego vehicle distance
            ego_corners = corners_cal(se_f1,le,we,corner_dir)
            ego_corners = ego_corners.T
            closest_point_dis2 = [ [tuple(corner)]+collision_dis_dir(corner,wo,lo) for corner in ego_corners]
            closest_point_dis2 = (Sort_List(closest_point_dis2))
        #     print('dis:',closest_point_dis2[0][2])
            
        #     if closest_point_dis2[0][2] <0 or closest_point_dis[0][2] <0:
        #         print(se,so)
            
            # print(closest_point_dis2)
            
            if closest_point_dis[0][2]<=closest_point_dis2[0][2]:
                #transfer back to original coordinates closest_point_dis[0][1]
                obstacle_point = point_transformation(se,closest_point_dis[0][0])
                ego_point = point_transformation(se,closest_point_dis[0][1])
                return (obstacle_point, ego_point, closest_point_dis[0][2])
            else:
                #transfer back to original coordinates closest_point_dis[0][1]
                obstacle_point = point_transformation(so,closest_point_dis2[0][1])
                ego_point = point_transformation(so,closest_point_dis2[0][0])
                

            
            
                #define Matplotlib figure and axis


            
            

            
            
            #display plot
        
            return (obstacle_point, ego_point, closest_point_dis2[0][2])



        

        # %%
        from scipy.stats import norm
        def collision_probablity(V, P, dis, obs_cov):
            PV = V - P
            VP = P - V
            theta_pv =  np.arccos(np.dot(PV,np.array([1,0]))/np.linalg.norm(PV))
            R = np.array([ [np.cos(theta_pv), -np.sin(theta_pv)],
                            [np.sin(theta_pv), np.cos(theta_pv)]])
        #     print(np.matmul(R, R.T))
            den = np.matmul(np.matmul(R, obs_cov),R.T)
        #     print(np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1])
            point_dis = np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1]
            col_prob =  norm.cdf( -dis/ np.sqrt(den[0,0]))

            return col_prob

        # Get collision points and distance
        obstacle_point, ego_point, distance = collision_point_rect(
            ego_state, 
            obstacle_state,
            we=self.ego_width,
            le=self.ego_length,
            wo=self.obs_width,
            lo=self.obs_length
        )
        
        if distance == -1:  # Already colliding
            return 1.0
            
        # Calculate collision probability
        col_prob = collision_probablity(
            np.array(obstacle_point),
            np.array(ego_point),
            distance,
            self.obs_cov
        )
        
        return col_prob
    
    def calculate_collision_probability_with_trajectory(self, ego_trajectory_point, obstacle_state):
        return self.calculate_collision_probability(ego_trajectory_point, obstacle_state)

class KFObstacleTracker(BaseTracker):
    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.kf = self._initialize_kalman_filter()
        self.obs_cov = np.identity(2) * 0.04
        
    def _initialize_kalman_filter(self):
        # State: [x, y, theta, vx, vy, omega], Measurement: [x, y, theta]
        kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # Use dt to construct F:
        kf.F = np.array([
            [1, 0, 0, self.dt,    0,      0],
            [0, 1, 0,     0, self.dt,      0],
            [0, 0, 1,     0,     0,  self.dt],
            [0, 0, 0,     1,     0,      0],
            [0, 0, 0,     0,     1,      0],
            [0, 0, 0,     0,     0,      1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # measure x
            [0, 1, 0, 0, 0, 0],  # measure y
            [0, 0, 1, 0, 0, 0]   # measure theta
        ])
        
        # Measurement noise
        kf.R = np.eye(3) * 0.1
        
        # Process noise
        kf.Q = np.eye(6) * 0.1
        
        # Initial state covariance
        kf.P *= 1000
        
        return kf
    
    def update(self, state, tick_number):
        """Update tracker with new state measurement (x, y, theta in radians)"""
        x, y, yaw_deg = state
        # Convert yaw to radians:
        theta = math.radians(yaw_deg)
        
        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)
        measurement = np.array([x, y, theta])
        self.kf.predict()
        self.kf.update(measurement)
        
    def predict_future_position(self, steps_ahead):
        """Predict future position using Kalman filter"""
        all_predicted_states = []
        state = self.kf.x.copy()
        for _ in range(steps_ahead):
            state = np.dot(self.kf.F, state)
            all_predicted_states.append([state[0][0], state[1][0], state[2][0]])
        return all_predicted_states

class GroundTruthTracker(BaseTracker):
    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05, trajectory_file='./obstacle_trajectory.csv'):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.trajectory_file = trajectory_file
        self.trajectory = self.load_trajectory()
        self.current_index = 0
        self.obs_cov = np.identity(2) * 0.001  # Very small uncertainty for ground truth
        
    def load_trajectory(self):
        """Load pre-recorded trajectory from file"""
        trajectory = []
        with open(self.trajectory_file, 'r') as f:
            for line in f:
                x, y, yaw = map(float, line.strip().split(','))
                trajectory.append([x, y, math.radians(yaw)])
        return trajectory
    
    def update(self, state, tick_number):
        """Update tracker with new state measurement"""
        x, y, yaw_deg = state
        theta = math.radians(yaw_deg)
        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)
        self.current_index = tick_number
    
    def predict_future_position(self, steps_ahead):
        """Predict future position using ground truth trajectory"""
        predicted_states = []
        for i in range(steps_ahead):
            future_index = self.current_index + i
            if future_index < len(self.trajectory):
                predicted_states.append(self.trajectory[future_index])
            else:
                # If we run out of trajectory, use the last known position
                predicted_states.append(self.trajectory[-1])
        return predicted_states

class EKFObstacleTracker(BaseTracker):
    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.state = np.zeros((5, 1))  # [x, y, theta, v, omega]
        self.P = np.eye(5) * 1000  # Initial state covariance
        self.Q = np.eye(5) * 0.1   # Process noise
        self.R = np.eye(3) * 0.1   # Measurement noise
        self.obs_cov = np.identity(2) * 0.04
        
    def _f(self, x, dt):
        """State transition function"""
        F = np.array([
            [x[0,0] + x[3,0] * np.cos(x[2,0]) * dt],  # x + v*cos(theta)*dt
            [x[1,0] + x[3,0] * np.sin(x[2,0]) * dt],  # y + v*sin(theta)*dt
            [x[2,0] + x[4,0] * dt],                    # theta + omega*dt
            [x[3,0]],                                  # v
            [x[4,0]]                                   # omega
        ])
        return F
    
    def _F(self, x, dt):
        """Jacobian of state transition function"""
        F = np.array([
            [1, 0, -x[3,0]*np.sin(x[2,0])*dt, np.cos(x[2,0])*dt, 0],
            [0, 1,  x[3,0]*np.cos(x[2,0])*dt, np.sin(x[2,0])*dt, 0],
            [0, 0,                         1,                  0, dt],
            [0, 0,                         0,                  1,  0],
            [0, 0,                         0,                  0,  1]
        ])
        return F
    
    def _h(self, x):
        """Measurement function"""
        H = np.array([
            [x[0,0]],  # x
            [x[1,0]],  # y
            [x[2,0]]   # theta
        ])
        return H
    
    def _H(self, x):
        """Jacobian of measurement function"""
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        return H
    
    def update(self, state, tick_number):
        """Update tracker with new state measurement"""
        x, y, yaw_deg = state
        theta = math.radians(yaw_deg)
        
        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)
        
        # Prediction step
        x_pred = self._f(self.state, self.dt)
        F = self._F(self.state, self.dt)
        self.P = F @ self.P @ F.T + self.Q
        
        # Update step
        z = np.array([[x], [y], [theta]])
        H = self._H(x_pred)
        y = z - self._h(x_pred)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = x_pred + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
        
    def predict_future_position(self, steps_ahead):
        """Predict future positions using EKF"""
        predicted_states = []
        current_state = self.state.copy()
        
        for _ in range(steps_ahead):
            current_state = self._f(current_state, self.dt)
            predicted_states.append([
                current_state[0,0],  # x
                current_state[1,0],  # y
                current_state[2,0]   # theta
            ])
            
        return predicted_states

def save_trajectory(vehicle, filename):
    """Save vehicle transform to file"""
    with open(filename, 'a') as f:
        transform = vehicle.get_transform()
        f.write(f"{transform.location.x},{transform.location.y},{transform.rotation.yaw}\n")

def load_trajectory(filename):
    """Load trajectory from file"""
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, yaw = map(float, line.strip().split(','))
            trajectory.append([x, y, math.radians(yaw)])
    return trajectory

def run_first_simulation(config, trajectory_file=None):
    """Run the first simulation to generate the ego vehicle trajectory"""
    # Use trajectory file from config if none provided
    if trajectory_file is None:
        trajectory_file = config['trajectories']['ego']
    
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    # Setup ego vehicle spawn point
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    # Spawn only the ego vehicle
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    ego_vehicle.set_autopilot(False)

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] + 
                         config['ego_vehicle']['turn_ticks'] + 
                         config['ego_vehicle']['after_turn_ticks']):
            world.tick()

            # Apply controls and save trajectory
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                # Initial straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks']:
                # Turning phase
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = 0.0
            
            ego_vehicle.apply_control(ego_control)
            save_trajectory(ego_vehicle, trajectory_file)

    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        client.reload_world()
        world.apply_settings(original_settings)

def run_simulation(config):
    """Run a simulation with the given configuration using pre-recorded trajectory"""
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    # Find two spawn points close to each other
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn the ego vehicle
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')


    # Spawn both vehicles
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)
    
    # Load the reference trajectory
    ego_trajectory = load_trajectory(config['trajectories']['ego'])
    tracker_type = "ekf"  # Options: "kf", "ekf", "ground_truth"
    if tracker_type == "ground_truth":
        obstacle_tracker = GroundTruthTracker(ego_vehicle, obstacle_vehicle, 
                                            dt=config['simulation']['delta_seconds'])
    elif tracker_type == "ekf":
        obstacle_tracker = EKFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                            dt=config['simulation']['delta_seconds'])
    else:  # "kf"
        obstacle_tracker = KFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                           dt=config['simulation']['delta_seconds'])

    # No autopilot, we will manually control both
    ego_vehicle.set_autopilot(False)
    if obstacle_vehicle is not None:
        obstacle_vehicle.set_autopilot(False)

    # Attach a top-down BEV camera above the intersection or ego vehicle’s start
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['video']['width']))
    camera_bp.set_attribute('image_size_y', str(config['video']['height']))
    camera_bp.set_attribute('fov', config['camera']['fov'])

    camera_transform = carla.Transform(
        carla.Location(x=ego_spawn_point.location.x + config['camera']['offset']['x'],
                       y=ego_spawn_point.location.y + config['camera']['offset']['y'],
                       z=config['camera']['height']),
        carla.Rotation(pitch=-90)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=None)

    # Only create video writer if save_video is True
    video_writer = None
    if config['save_options']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            config['video']['filename'], 
            fourcc, 
            config['video']['fps'], 
            (config['video']['width'], config['video']['height'])
        )

    collision_prob = 0.0
    frame_queue = []
    def camera_callback(image):
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()
        
        frame_queue.append(frame_bgr)
        
        # Only save individual frames if save_images is True
        if config['save_options']['save_images']:
            cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

    camera.listen(camera_callback)

    # Initialize obstacle tracker with dt from simulation settings
    obstacle_buffer = [
        obstacle_vehicle.get_transform()
    ] * config['simulation']['l_max']

    # Add collision sensor
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
    
    # Collision flag
    has_collided = False
    
    def collision_callback(event):
        nonlocal has_collided
        has_collided = True
        # print(f"Collision detected with {event.other_actor}")
    
    collision_sensor.listen(collision_callback)

    # Add CSV setup for collision probability logging
    collision_prob_file = './collision_probabilities.csv'
    
    if os.path.exists(collision_prob_file):
        pass 
    else:   
        with open(collision_prob_file, 'w') as f:
            f.write('timestamp,tick,delta_k,collision_probability\n')

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks'] + config['ego_vehicle']['after_turn_ticks']):
            world.tick()
            
            # Check for collision
            if has_collided:
                print("Collision detected! Stopping vehicles.")
                # Stop both vehicles
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if obstacle_vehicle is not None:
                    obstacle_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break

            # Update buffer with current transform
            current_transform = obstacle_vehicle.get_transform()
            obstacle_buffer.append(current_transform)
            obstacle_buffer.pop(0)
            
            # Only update obstacle tracking at delta_k intervals
            # if tick % config['simulation']['delta_k'] == 0:
            # Get historical transform from l_max steps ago
            historical_transform = obstacle_buffer[0]  # oldest transform in buffer
            
            obstacle_tracker.update(
                (
                    historical_transform.location.x,
                    historical_transform.location.y,
                    historical_transform.rotation.yaw
                ),
                tick
            )
            
            predicted_ego_positions = obstacle_tracker.predict_future_position(
                int(config['simulation']['prediction_steps'] / config['simulation']['delta_k'])
            )
            collision_probabilities = []
            
            for step, predicted_pos in enumerate(predicted_ego_positions):
                if tick + step < len(ego_trajectory):
                    ego_trajectory_point = ego_trajectory[tick + step]
                    predicted_pos = [predicted_pos[0], predicted_pos[1], predicted_pos[2]]
                    collision_prob = obstacle_tracker.calculate_collision_probability_with_trajectory(
                        ego_trajectory_point,
                        predicted_pos
                    )
                    collision_probabilities.append(collision_prob)
            
            # print (predicted_ego_positions, collision_probabilities) mapping 
            
            collision_prob = max(collision_probabilities)
            collision_time = collision_probabilities.index(collision_prob)
            
            # print(f"Tick {tick}: Max collision probability: {collision_prob:.4f} at time step {collision_time}")
            
            # Vehicle control logic remains the same
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                # Initial straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks']:
                # Turning phase
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = 0.0
            
            ego_vehicle.apply_control(ego_control)
            
            obstacle_control = carla.VehicleControl()
            obstacle_control.throttle = config['obstacle_vehicle']['throttle']
            obstacle_control.steer = config['obstacle_vehicle']['steer']
            obstacle_vehicle.apply_control(obstacle_control)

            # Write queued frames to video
            while frame_queue:
                frame_bgr = frame_queue.pop(0)
                
                # Add collision probability text to the frame
                collision_text = f"Collision Probability: {collision_prob:.4f}"
                cv2.putText(frame_bgr, collision_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if config['save_options']['save_video']:
                    video_writer.write(frame_bgr)
                if config['save_options']['save_images']:
                    cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

            # After calculating collision_prob and collision_time
            timestamp = tick * config['simulation']['delta_seconds']
            with open(collision_prob_file, 'a') as f:
                f.write(f'{timestamp:.2f},{tick},{config["simulation"]["delta_k"]},{collision_prob:.4f}\n')

    finally:
        # Add collision sensor cleanup
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()
            
        # Cleanup
        camera.stop()
        if video_writer is not None:
            video_writer.release()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        if camera is not None:
            camera.destroy()

        client.reload_world()
        world.apply_settings(original_settings)
        # print(f"Video saved to {config['video']['filename']}")


def run_adaptive_simulation(config):
    """Run a simulation with adaptive delta_k and braking based on collision probability"""
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    # Setup ego vehicle spawn point
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn both vehicles
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    # Attach camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['video']['width']))
    camera_bp.set_attribute('image_size_y', str(config['video']['height']))
    camera_bp.set_attribute('fov', config['camera']['fov'])

    camera_transform = carla.Transform(
        carla.Location(x=ego_spawn_point.location.x + config['camera']['offset']['x'],
                       y=ego_spawn_point.location.y + config['camera']['offset']['y'],
                       z=config['camera']['height']),
        carla.Rotation(pitch=-90)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=None)

    # Video writer setup
    video_writer = None
    if config['save_options']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            config['video']['filename'], 
            fourcc, 
            config['video']['fps'], 
            (config['video']['width'], config['video']['height'])
        )

    frame_queue = []
    def camera_callback(image):
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()
        
        frame_queue.append(frame_bgr)
        
        if config['save_options']['save_images']:
            cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

    camera.listen(camera_callback)

    # Load trajectory and setup tracker
    ego_trajectory = load_trajectory(config['trajectories']['ego'])
    
    # Initialize tracker with initial delta_k
    initial_delta_k = config['simulation']['delta_k']
    current_delta_k = initial_delta_k
    
    if config['simulation']['tracker_type'] == 'ekf':   
        obstacle_tracker = EKFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                            dt=config['simulation']['delta_seconds'])
    else:
        obstacle_tracker = KFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                           dt=config['simulation']['delta_seconds'])

    # Add ground truth tracker (using same type as regular tracker)
    if config['simulation']['tracker_type'] == 'ekf':   
        ground_truth_tracker = EKFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                                dt=config['simulation']['delta_seconds'])
    else:
        ground_truth_tracker = KFObstacleTracker(ego_vehicle, obstacle_vehicle, 
                                               dt=config['simulation']['delta_seconds'])

    # Initialize obstacle buffer
    obstacle_buffer = []

    # Add collision sensor
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
    
    has_collided = False
    def collision_callback(event):
        nonlocal has_collided
        has_collided = True
        # print(f"Collision detected with {event.other_actor}")
    
    collision_sensor.listen(collision_callback)

    # Setup CSV logging
    collision_prob_file = './collision_probabilities.csv'
    if not os.path.exists(collision_prob_file):
        with open(collision_prob_file, 'w') as f:
            f.write('timestamp,tick,delta_k,collision_probability,ground_truth_probability\n')

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] + 
                         config['ego_vehicle']['turn_ticks'] + 
                         config['ego_vehicle']['after_turn_ticks']):
            
            world.tick()

            if has_collided:
                # print("Collision detected! Stopping vehicles.")
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                obstacle_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break

            current_transform = obstacle_vehicle.get_transform()
            ground_truth_tracker.update(
                (
                    current_transform.location.x,
                    current_transform.location.y,
                    current_transform.rotation.yaw
                ),
                tick
            )

            obstacle_buffer.append(current_transform)
            brake = False
            max_collision_prob = 0.0
            ground_truth_collision_prob = 0.0
            
            if tick >= config['simulation']['l_max']:
                obstacle_buffer.pop(0)
            
                historical_transform = obstacle_buffer[0]
                
                obstacle_tracker.update(
                    (
                        historical_transform.location.x,
                        historical_transform.location.y,
                        historical_transform.rotation.yaw
                    ),
                    tick
                )
                
                # Predict future positions and calculate collision probabilities
                predicted_positions = obstacle_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] / current_delta_k)
                )
                
                max_collision_prob, collision_time, collision_probabilities = calculate_collision_probabilities(
                    obstacle_tracker,
                    predicted_positions,
                    ego_trajectory,
                    tick
                )
                ground_truth_predictions = ground_truth_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] / current_delta_k)
                )

                ground_truth_max_prob, ground_truth_collision_time, ground_truth_probabilities = calculate_collision_probabilities(
                    ground_truth_tracker,
                    ground_truth_predictions,
                    ego_trajectory,
                    tick
                )
                # Get current ego vehicle position
                ego_transform = ego_vehicle.get_transform()
                ego_pos = (ego_transform.location.x, ego_transform.location.y, math.radians(ego_transform.rotation.yaw))
                obstacle_pos = (current_transform.location.x, current_transform.location.y, math.radians(current_transform.rotation.yaw))
                max_prob_idx = collision_probabilities.index(max_collision_prob)
                predicted_pos = predicted_positions[max_prob_idx]
                ego_predicted_pos =  ego_trajectory[tick + max_prob_idx]
                ground_truth_collision_prob = ground_truth_max_prob
                # Get predicted position with highest collision probability
                # print(f"\nTick {tick}:")
                # print("obstacle vehicle current position:")
                # print(f"(x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f}, yaw={obstacle_pos[2]:.2f})")
                # print(f"Predicted obstacle position with max collision prob {max_collision_prob:.4f}:")
                # print(f"(x={predicted_pos[0]:.2f}, y={predicted_pos[1]:.2f}, yaw={predicted_pos[2]:.2f})")
                # # ego vehicle predicted position
                # print(f"Current ego position: (x={ego_pos[0]:.2f}, y={ego_pos[1]:.2f}, yaw={ego_pos[2]:.2f})")
                
                # print(f"Ego vehicle predicted position:")
                # print(f"(x={ego_predicted_pos[0]:.2f}, y={ego_predicted_pos[1]:.2f}, yaw={ego_predicted_pos[2]:.2f})")
                # print(f"Groundtruth collision probability:{ground_truth_collision_prob:.4f}")
                
                # Adaptive behavior based on collision probability
                if max_collision_prob > config['simulation']['emergency_brake_threshold']:
                    # Emergency brake
                    brake = True
                    print(f"Emergency brake activated! Collision probability: {max_collision_prob:.4f}")
                        
                elif max_collision_prob > config['simulation']['cautious_threshold']:
                    # Increase tracking frequency (decrease delta_k)
                    new_delta_k = config['simulation']['cautious_delta_k']
                    if new_delta_k != current_delta_k:
                        print(f"Adjusting delta_k from {current_delta_k} to {new_delta_k}")
                        current_delta_k = new_delta_k
                        # drop the obstacle buffer to new delta_k
                        for i in range(config['simulation']['l_max'] - new_delta_k):
                            obstacle_pos = obstacle_buffer.pop(0)
                            # update obstacle tracker with the new position
                            obstacle_tracker.update(
                                (
                                    obstacle_pos.location.x,
                                    obstacle_pos.location.y,
                                    obstacle_pos.rotation.yaw
                                ),
                                tick
                            )
                
            if brake:
                ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
            else:
                # Normal driving
                ego_control = carla.VehicleControl()
                if tick < config['ego_vehicle']['go_straight_ticks']:
                    ego_control.throttle = config['ego_vehicle']['throttle']['straight']
                elif tick < config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks']:
                    ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                    ego_control.steer = config['ego_vehicle']['steer']['turn']
                else:
                    ego_control.throttle = config['ego_vehicle']['throttle']['after_turn']
            
            ego_vehicle.apply_control(ego_control)
            
            # Apply controls
            obstacle_control = carla.VehicleControl()
            
            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                # Initial straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config['obstacle_vehicle']['turn_ticks']:
                # Turning phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['after_turn']
                
            obstacle_vehicle.apply_control(obstacle_control)

            # Process camera frames
            while frame_queue:
                frame_bgr = frame_queue.pop(0)
                
                # Add collision probability text to the frame
                collision_text = f"Predicted Collision Probability: {max_collision_prob:.4f}"
                cv2.putText(frame_bgr, collision_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Add ground truth collision probability text to the frame
                ground_truth_collision_text = f"Groundtruth Collision Probability: {ground_truth_collision_prob:.4f}"
                cv2.putText(frame_bgr, ground_truth_collision_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                # Add delta_k text to the frame
                delta_k_text = f"Current Latency: {current_delta_k * 10} ms"
                cv2.putText(frame_bgr, delta_k_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                
                if config['save_options']['save_video']:
                    video_writer.write(frame_bgr)
                if config['save_options']['save_images']:
                    cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

            # Log data
            timestamp = tick * config['simulation']['delta_seconds']
            with open(collision_prob_file, 'a') as f:
                f.write(f'{timestamp:.2f},{tick},{current_delta_k},{max_collision_prob:.4f},{ground_truth_collision_prob:.4f}\n')

    finally:
        # Cleanup
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()
            
        camera.stop()
        if video_writer is not None:
            video_writer.release()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        if camera is not None:
            camera.destroy()

        client.reload_world()
        world.apply_settings(original_settings)
        # print(f"Video saved to {config['video']['filename']}")
        return has_collided, current_delta_k
        

def run_obstacle_only_simulation(config, trajectory_file=None):
    """Run a simulation with only the obstacle vehicle to record its trajectory"""
    # Use trajectory file from config if none provided
    if trajectory_file is None:
        trajectory_file = config['trajectories']['obstacle']
    
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    # Setup obstacle vehicle spawn point based on ego vehicle's position
    # (since obstacle spawn is defined relative to ego in config)
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn only the obstacle vehicle
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    if obstacle_vehicle is None:
        raise RuntimeError("Failed to spawn obstacle vehicle")

    try:
        # Run for the same duration as the full simulation
        total_ticks = (config['ego_vehicle']['go_straight_ticks'] + 
                      config['ego_vehicle']['turn_ticks'] + 
                      config['ego_vehicle'].get('after_turn_ticks', 0))

        for tick in range(total_ticks):
            world.tick()

            # Apply phase-based control to obstacle vehicle
            obstacle_control = carla.VehicleControl()
            
            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                # Initial straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config['obstacle_vehicle']['turn_ticks']:
                # Turning phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['after_turn']
                
            obstacle_vehicle.apply_control(obstacle_control)
            save_trajectory(obstacle_vehicle, trajectory_file)

    finally:
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        
        # Restore original settings
        world.apply_settings(original_settings)
        print(f"Obstacle trajectory saved to {trajectory_file}")

def calculate_collision_probabilities(obstacle_tracker, predicted_positions, ego_trajectory, tick):
    """
    Calculate collision probabilities for predicted positions against ego trajectory.
    
    Args:
        obstacle_tracker: The tracker object used for collision probability calculation
        predicted_positions: List of predicted future positions of the obstacle
        ego_trajectory: List of ego vehicle trajectory points
        tick: Current simulation tick
    
    Returns:
        tuple: (max_collision_prob, collision_time, collision_probabilities)
            - max_collision_prob: Maximum collision probability across all predictions
            - collision_time: Time step at which maximum collision probability occurs
            - collision_probabilities: List of all calculated collision probabilities
    """
    collision_probabilities = []
    for step, predicted_pos in enumerate(predicted_positions):
        if tick + step < len(ego_trajectory):
            ego_trajectory_point = ego_trajectory[tick + step]
            predicted_pos = [predicted_pos[0], predicted_pos[1], predicted_pos[2]]
            collision_prob = obstacle_tracker.calculate_collision_probability_with_trajectory(
                ego_trajectory_point,
                predicted_pos
            )
            collision_probabilities.append(collision_prob)
    
    max_collision_prob = max(collision_probabilities) if collision_probabilities else 0.0
    collision_time = collision_probabilities.index(max_collision_prob) if collision_probabilities else 0
    
    print(f"Tick {tick}: Max collision probability: {max_collision_prob:.4f} at time step {collision_time}")
    
    return max_collision_prob, collision_time, collision_probabilities

def get_monte_carlo_spawn_point(config, ego_spawn_point, std_dev=1):
    """
    Generate a randomized spawn point for the obstacle vehicle using Monte Carlo sampling.
    
    Args:
        config: Configuration dictionary
        ego_spawn_point: Base ego vehicle spawn point
        std_dev: Standard deviation for the normal distribution (in meters)
    
    Returns:
        carla.Transform: Randomized spawn point
    """
    base_x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    base_y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    base_yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']
    
    # Sample from normal distribution for position and yaw
    x = np.random.normal(base_x, std_dev)
    y = np.random.normal(base_y, std_dev)
    yaw = np.random.normal(base_yaw, std_dev * 2)  # Larger variation in orientation
    
    spawn_point = carla.Transform(
        carla.Location(x=x, y=y, z=0.0),
        carla.Rotation(yaw=yaw)
    )
    
    
    return spawn_point

def run_monte_carlo_simulation(config, num_samples=10):
    """
    Run multiple simulations with Monte Carlo sampling of obstacle spawn points.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of Monte Carlo samples to run
    
    Returns:
        dict: Statistics about collisions and spawn points
    """
    collision_stats = {
        'num_collisions': 0,
        'spawn_points': [],
        'collision_cases': []
    }
    
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")
    
    # Get base spawn point for ego vehicle
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']
    
    for sample in range(num_samples):
        print(f"\nRunning Monte Carlo sample {sample + 1}/{num_samples}")
        
        # Generate random spawn point
        obstacle_spawn_point = get_monte_carlo_spawn_point(config, ego_spawn_point)
        
        # Update config with new spawn point
        sample_config = dict(config)
        sample_config['obstacle_vehicle']['spawn_offset'] = {
            'x': obstacle_spawn_point.location.x - ego_spawn_point.location.x,
            'y': obstacle_spawn_point.location.y - ego_spawn_point.location.y,
            'yaw': obstacle_spawn_point.rotation.yaw - ego_spawn_point.rotation.yaw
        }
        
        # Update trajectory files for this sample
        sample_config['trajectories'] = {
            'ego': f'./ego_trajectory_sample_{sample}.csv',
            'obstacle': f'./obstacle_trajectory_sample_{sample}.csv'
        }
        try:
            # Generate obstacle trajectory
            run_obstacle_only_simulation(sample_config)
        except Exception as e:
            # obstacle spawn point is invalid
            print(f"Error in sample {sample}: {e}")
            continue
        
        try:
            # Generate trajectories
            run_first_simulation(sample_config)
            
            # Run simulation and check for collision
            has_collided, current_delta_k = run_adaptive_simulation(sample_config)
            
            # Check if collision occurred
            if has_collided:
                collision_stats['num_collisions'] += 1
                collision_stats['collision_cases'].append({
                    'spawn_point': obstacle_spawn_point,
                    'sample_num': sample
                })
            else:
                collision_stats['spawn_points'].append(obstacle_spawn_point)
            
        except Exception as e:
            print(f"Error in sample {sample}: {e}")
            collision_stats['num_collisions'] += 1
            collision_stats['collision_cases'].append({
                'spawn_point': obstacle_spawn_point,
                'sample_num': sample,
                'error': str(e)
            })
        
        # Clean up trajectory files
        for trajectory_file in sample_config['trajectories'].values():
            if os.path.exists(trajectory_file):
                os.remove(trajectory_file)
                
        if not os.path.exists('./monte_carlo_results/statistics.csv'):
            with open('./monte_carlo_results/statistics.csv', 'w') as f:
                f.write(f"scenario,lmax,delta_k,collision,delta_k_used\n")
        # save and append the stats to a csv file
        with open('./monte_carlo_results/statistics.csv', 'a') as f:
            # scenario, lmax, delta_k, collision yes or no, delta_k_used
            scenario_type = config['video']['filename'].split('/')[-1].split('_collision')[0]
            f.write(f"{scenario_type},{config['simulation']['l_max']},{current_delta_k},{has_collided},{config['simulation']['delta_k']}\n")
        
    return collision_stats

def restart_carla_docker():
    """Restart the CARLA Docker container"""
    import subprocess
    import time
    
    try:
        # Stop existing CARLA container
        subprocess.run(['docker', 'stop', 'carla'], check=False)
        subprocess.run(['docker', 'rm', 'carla'], check=False)
        
        # Start new CARLA container
        subprocess.run([
            'docker', 'run', '-d',
            '--name=carla',
            '--privileged',
            '--gpus', 'all',
            '--net=host',
            '-v', '/tmp/.X11-unix:/tmp/.X11-unix:rw',
            'carlasim/carla:0.9.15',
            '/bin/bash', './CarlaUE4.sh', '-RenderOffScreen'
        ], check=True)
        
        # Wait for CARLA to initialize
        time.sleep(10)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restarting CARLA container: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run CARLA simulation with configurable parameters')
    parser.add_argument('--cautious_delta_k', type=int, default=-1,
                      help='Value for cautious_delta_k parameter')
    parser.add_argument('--config_type', type=str, choices=['right_turn', 'left_turn', 'merge'],
                      default='merge', help='Type of configuration to use')
    parser.add_argument('--emergency_brake_threshold', type=float, default=1.1,
                      help='Threshold for emergency braking')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to store results')
    args = parser.parse_args()
    
    # Select configuration based on argument
    config_map = {
        'right_turn': unprotected_right_turn_config,
        'left_turn': unprotected_left_turn_config,
        'merge': opposite_direction_merge_config
    }
    
    base_config = config_map[args.config_type]
    
    # Update configuration with command line parameters
    if args.cautious_delta_k != -1:
        base_config['simulation']['cautious_delta_k'] = args.cautious_delta_k
        base_config['simulation']['l_max'] = args.cautious_delta_k
        base_config['simulation']['delta_k'] = args.cautious_delta_k
    
    # Update emergency brake threshold
    base_config['simulation']['emergency_brake_threshold'] = args.emergency_brake_threshold
    
    # Update output paths based on output directory
    base_config['video']['filename'] = os.path.join(args.output_dir, 'simulation.mp4')
    base_config['trajectories']['ego'] = os.path.join(args.output_dir, 'ego_trajectory.csv')
    base_config['trajectories']['obstacle'] = os.path.join(args.output_dir, 'obstacle_trajectory.csv')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bev_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'monte_carlo_results'), exist_ok=True)
    
    max_retries = 3
    retry_count = 0
    
    # Clean old files in the output directory
    if os.path.exists(os.path.join(args.output_dir, 'bev_images')):
        shutil.rmtree(os.path.join(args.output_dir, 'bev_images'))
    os.makedirs(os.path.join(args.output_dir, 'bev_images'))
    
    collision_prob_file = os.path.join(args.output_dir, 'collision_probabilities.csv')
    if os.path.exists(collision_prob_file):
        os.remove(collision_prob_file)
    
    # Base configuration for Monte Carlo simulation
    num_samples = 1
    
    while retry_count < max_retries:
        try:
            # Run Monte Carlo simulation
            stats = run_monte_carlo_simulation(base_config, num_samples)
            
            # Calculate and save statistics
            collision_rate = stats['num_collisions'] / num_samples
            
            # Save results
            stats_file = os.path.join(args.output_dir, 'monte_carlo_results', 'statistics.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Monte Carlo Simulation Results\n")
                f.write(f"Configuration:\n")
                f.write(f"  Config Type: {args.config_type}\n")
                f.write(f"  Cautious Delta K: {base_config['simulation']['cautious_delta_k']}\n")
                f.write(f"  Emergency Brake Threshold: {base_config['simulation']['emergency_brake_threshold']}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Number of samples: {num_samples}\n")
                f.write(f"  Number of collisions: {stats['num_collisions']}\n")
                f.write(f"  Collision rate: {collision_rate:.2%}\n\n")
                
                f.write("Collision cases:\n")
                for case in stats['collision_cases']:
                    f.write(f"Sample {case['sample_num']}:\n")
                    f.write(f"  Spawn point: x={case['spawn_point'].location.x:.2f}, "
                           f"y={case['spawn_point'].location.y:.2f}, "
                           f"yaw={case['spawn_point'].rotation.yaw:.2f}\n")
                    if 'error' in case:
                        f.write(f"  Error: {case['error']}\n")
            
            print(f"\nMonte Carlo simulation completed.")
            print(f"Collision rate: {collision_rate:.2%} ({stats['num_collisions']}/{num_samples} collisions)")
            print(f"Results saved to {args.output_dir}")
            break
            
        except Exception as e:
            retry_count += 1
            print(f"\nError occurred: {e}")
            print(f"Retry {retry_count}/{max_retries}")
            
            print("CARLA connection issue detected. Attempting to restart CARLA...")
            if restart_carla_docker():
                print("CARLA successfully restarted")
                time.sleep(5)  # Give additional time for CARLA to stabilize
            else:
                print("Failed to restart CARLA")
              
            if retry_count >= max_retries:
                print("Max retries reached. Exiting...")
                break
            
            time.sleep(5)  # Wait before retrying

if __name__ == '__main__':
    main()
