import rclpy 
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math
import time
import pickle
import numpy as np
import uuid

from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from nav_msgs.msg import Path

from .trajectory_generation.rrt_algorithms.rrt.rrt import RRT
from .trajectory_generation.rrt_algorithms.rrt.rrt_star import RRTStar
from .trajectory_generation.rrt_algorithms.rrt.rrt_connect import RRTConnect
from .trajectory_generation.rrt_algorithms.search_space.search_space import SearchSpace
from .trajectory_generation.rrt_algorithms.utilities.plotting import Plot
from .trajectory_generation.rrt_algorithms.utilities.geometry import steer
class VehicleCommandMapping: 
    VEHICLE_CMD_NAV_LAND = 21
    VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
    VEHICLE_CMD_DO_SET_MODE = 176


class GlobalPlanner(Node): 
    def __init__(self) -> None: 
        super().__init__('global_planner')

        ### Parameters ###########################################
        self.mode = 'RRT_star'  

        # Octomap 
        self.octomap_resolution = 1.0

        # RRT
        self.RRT_search_space_range_x = (-1, 15)
        self.RRT_search_space_range_y = (-7, 7)
        self.RRT_search_space_range_z = (0, 3)
        self.RRT_goal = (13, 7, 1)
        self.RRT_initial = (0, 0, 1)
        self.RRT_q = 0.3  # length of tree edges
        self.RRT_r = 1  # length of smallest edge to check for intersection with obstacles
        self.RRT_max_samples = 3000  # max number of samples to take before timing out
        self.RRT_prc = 0.1  # probability of checking for a connection to goal
        # RRT*
        self.RRT_rewire_count = 32  # optional, number of nearby branches to rewire

        self.save_pc2_octomap = True

        ###########################################################

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        ### Subscribers ###
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        
        self.vehicle_command_subscriber = self.create_subscription(
            VehicleCommand, "/fmu/in/vehicle_command", self.vehicle_command_callback, qos_profile)
        
        self.octomap_pc2_sub = self.create_subscription(
            PointCloud2, '/octomap_point_cloud_centers', self.octomap_pc2_callback, 10)

        ### Publishers ###
        self.waypoint_publisher = self.create_publisher(Vector3Stamped, "/global_waypoint", qos_profile)

        self.pc2_pub = self.create_publisher(PointCloud2, 'read_pc', 10)

        self.vehicle_path_pub = self.create_publisher(Path, '/RRT_path', 10)
        self.vehicle_path_msg = Path()

        ### Timers ###

        ### Initialize variables ###
        self.timer = self.create_timer(0.1, self.waypoint_callback)

        # PX4 commands
        self.command_mapping = {
            21: "VEHICLE_CMD_NAV_LAND", 
            400: "VEHICLE_CMD_COMPONENT_ARM_DISARM", 
            176: "VEHICLE_CMD_DO_SET_MODE",
        }
        
        # Octomap 
        self.octomap_occupied_pointcloud = []
        self.octomap_received = False

        self.vehicle_position = [0.0, 0.0, 0.0]

        # Trajectory
        self.trajectory_waypoints = [self.RRT_initial]
        self.wayp_idx = 0
        self.RRT_solved = False
        self.goal_reached = False

        self.get_logger().info("Initializing RRT search space")
        X_dimensions = np.array([self.RRT_search_space_range_x, self.RRT_search_space_range_y, self.RRT_search_space_range_z])
        # Create search space
        self.X = SearchSpace(X_dimensions)

        self.logger = self.get_logger()
        self.logger.info("Global planner node initialized")

    def octomap_pc2_callback(self, pointcloud: PointCloud2) -> None:
        self.get_logger().info("Octomap pointcloud received")
        
        for p in point_cloud2.read_points(pointcloud, field_names = ("x", "y", "z"), skip_nans=True):
            # Get XYZ coordinates to calculate vertical angle and filter by vertical scans
            x = p[0]
            y = p[1]
            z = p[2]

            self.octomap_occupied_pointcloud.append([x, y, z])
        
        if self.save_pc2_octomap: 
            with open("pc2_octomap_list", "wb") as fp:   #Pickling
                pickle.dump(self.octomap_occupied_pointcloud, fp)

            self.save_pc2_octomap = False
        
        """ pc2_cropped = PointCloud2()
        pc2_cropped.header = pointcloud.header
        pc2_cropped.height = 1
        pc2_cropped.width = len(self.octomap_occupied_pointcloud)
        pc2_cropped.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.is_bigendian = False
        pc2_cropped.point_step = 12  # 4 (x) + 4 (y) + 4 (z) bytes per point
        pc2_cropped.row_step = pc2_cropped.point_step * len(self.octomap_occupied_pointcloud)
        pc2_cropped.data = np.array(self.octomap_occupied_pointcloud, dtype=np.float32).tobytes()
        self.pc2_pub.publish(pc2_cropped) """

        self.octomap_received = True


    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        # Convert NED (px4) to ENU (ROS)
        x = vehicle_local_position.y
        y = vehicle_local_position.x
        z = -vehicle_local_position.z

        self.vehicle_position = [x, y, z]

    def vehicle_command_callback(self, msg): 
        self.vehicle_command = msg.command
        self.vehicle_command_p1 = msg.param1

        if self.vehicle_command in self.command_mapping: 
            self.logger.info(f"vehicle command msg received: {self.vehicle_command}")
            if self.vehicle_command == 400: 
                if self.vehicle_command_p1 == 1.0: 
                    self.logger.info("Arming")
                elif self.vehicle_command_p1 == 0.0: 
                    self.logger.info("Disarming")
            elif self.vehicle_command == 21: 
                self.logger.info("Landing")

    def waypoint_callback(self): 
        # When octomap received, calculate RRT
        if self.octomap_received and not self.RRT_solved: 
            initial_time = time.time()
            self.get_logger().info("Inserting octomap as RRT obstacles")
            # Convert octomap occupied pointcloud in obstacles
            self.obstacles = self.octomap_pc2_to_obstacle()
            # Insert obstacles in search space
            for obstacle in self.obstacles: 
                self.X.obs.insert(uuid.uuid4().int, tuple(obstacle), tuple(obstacle))
            rrt = RRT(self.X, self.RRT_q, self.RRT_initial, self.RRT_goal, self.RRT_max_samples, self.RRT_r, self.RRT_prc)
            path = rrt.rrt_search()
            final_time = time.time()
            self.get_logger().info(f"RRT solved in {(final_time-initial_time)*1000} ms")
            self.get_logger().info(f"Path RRT: {path} ")
            self.octomap_received = False

            self.vehicle_path_msg.header.frame_id = 'livox_frame'
            self.vehicle_path_msg.header.stamp = self.get_clock().now().to_msg()

            self.n_waypoints = len(path)
            self.get_logger().info(f"Initial RRT wayps: {self.n_waypoints}")

            # Save the waypoints in trajectory_waypoints
            for idx, waypoint in enumerate(path): 
                if idx < (self.n_waypoints - 1):
                    pose_msg = self.vector2PoseMsg('odom', waypoint)
                    self.vehicle_path_msg.poses.append(pose_msg)
                    self.trajectory_waypoints.append(waypoint)
                    dist_next = np.sqrt((path[idx][0]-path[idx+1][0])**2 + 
                                        (path[idx][1]-path[idx+1][1])**2 +
                                        (path[idx][2]-path[idx+1][2])**2)
                    self.get_logger().info(f"Next wayp dist: {dist_next}")

            penultimate_waypoint = path[-2]
            last_waypoint = path[-1]
            last_distance = None
            
            # steer waypoints from last edge to not be farther than a threshold
            while last_distance is None or last_distance > self.RRT_q: 
                steered_wayp = steer(penultimate_waypoint, last_waypoint, self.RRT_q)
                last_distance = np.sqrt((steered_wayp[0]-last_waypoint[0])**2 + 
                                            (steered_wayp[1]-last_waypoint[1])**2 +
                                            (steered_wayp[2]-last_waypoint[2])**2
                                             )
                print(f"last_dist: {last_distance}")
                pose_msg = self.vector2PoseMsg('odom', steered_wayp)
                self.vehicle_path_msg.poses.append(pose_msg)
                if last_distance > self.RRT_q:
                    penultimate_waypoint = steered_wayp
                    self.trajectory_waypoints.append(penultimate_waypoint)
                else: 
                    self.trajectory_waypoints.append(last_waypoint)
            self.n_waypoints = len(self.trajectory_waypoints)
            self.get_logger().info(f"Steered Path RRT: {self.trajectory_waypoints}")
            
            self.vehicle_path_pub.publish(self.vehicle_path_msg)
            self.RRT_solved = True

        elif self.RRT_solved and self.goal_reached == False: 
            target_distance = self.distance_to_target(self.wayp_idx)
            self.get_logger().info(f"Distance to next waypoint {target_distance} m. {self.wayp_idx} idx")

            if target_distance < 0.1: 
                if self.wayp_idx == (self.n_waypoints - 1): 
                    a = 1
                    #self.wayp_idx = 0
                    #self.RRT_solved = False
                    #self.goal_reached = True
                else:
                    self.wayp_idx += 1

            target_waypoint = self.trajectory_waypoints[self.wayp_idx]
            wayp_msg = self.create_waypoint_msg(target_waypoint[0], target_waypoint[1], target_waypoint[2])
            self.vehicle_path_pub.publish(self.vehicle_path_msg)
            self.waypoint_publisher.publish(wayp_msg)

    def create_waypoint_msg(self, x, y, z):
        waypoint_msg = Vector3Stamped()

        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.header.frame_id = 'base_link'

        waypoint_msg.vector.x = float(x)
        waypoint_msg.vector.y = float(y)
        waypoint_msg.vector.z = float(z) 
        
        return waypoint_msg
    
    def distance_to_target(self, target_waypoint_idx):
        """
        Calculate the distance between the vehicle position and target waypoint

        Args:
            target_waypoint_idx (int): Index of the taregt waypoint

        Returns:
            float: The distance between the vehicle position and the specified vertex.
        """
        # Ensure valid vertex index
        if target_waypoint_idx < 0 or target_waypoint_idx == self.n_waypoints:
            self.logger.error("Invalid vertex index. It must be between 0 and 3.")
            return None

        # Get coordinates of the specified vertex
        target_waypoint = self.trajectory_waypoints[target_waypoint_idx]

        # Calculate distance using Euclidean distance formula
        dx = self.vehicle_position[0] - target_waypoint[0]
        dy = self.vehicle_position[1] - target_waypoint[1]
        dz = self.vehicle_position[2] - target_waypoint[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        return distance

    def octomap_pc2_to_obstacle(self): 
        obstacles = []
        for point in self.octomap_occupied_pointcloud:
            x1, y1, z1 = [coord - self.octomap_resolution/2 for coord in point]
            x2, y2, z2 = [coord + self.octomap_resolution/2 for coord in point]
            obstacle = [x1, y1, z1, x2, y2, z2]
            # TODO: Remove, just for plotting obstacles and remove ground
            if z1 > 0.2:
                obstacles.append(obstacle)

        return obstacles

    def vector2PoseMsg(self, frame_id, position):
        pose_msg = PoseStamped()
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        pose_msg.header.frame_id=frame_id
        """ pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3] """
        self.get_logger().info(f"Wayp: {position}")
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        return pose_msg

def main(): 
    rclpy.init()

    global_planner = GlobalPlanner()

    rclpy.spin(global_planner)

    global_planner.destroy_node()
    rclpy.shutdown()