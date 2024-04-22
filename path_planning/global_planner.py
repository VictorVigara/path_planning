import rclpy 
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math
import time
import pickle
import numpy as np

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

from .trajectory_generation.square_trajectory import square_vertices

class VehicleCommandMapping: 
    VEHICLE_CMD_NAV_LAND = 21
    VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
    VEHICLE_CMD_DO_SET_MODE = 176


class GlobalPlanner(Node): 
    def __init__(self) -> None: 
        super().__init__('global_planner')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        
        self.vehicle_command_subscriber = self.create_subscription(
            VehicleCommand, "/fmu/in/vehicle_command", self.vehicle_command_callback, qos_profile)
        
        self.octomap_pc2_sub = self.create_subscription(
            PointCloud2, '/octomap_point_cloud_centers', self.octomap_pc2_callback, 10)

        # Publishers
        self.waypoint_publisher = self.create_publisher(Vector3Stamped, "/global_waypoint", qos_profile)

        self.pc2_pub = self.create_publisher(PointCloud2, 'read_pc', 10)

        ### Parameters ###
        self.mode = 'square'  # 'square' sends a square trajectory

        # Square traj params
        self.square_side = 4.0
        self.square_height = 1.0

        self.save_pc2_octomap = True

        # Initialize variables
        self.command_mapping = {
            21: "VEHICLE_CMD_NAV_LAND", 
            400: "VEHICLE_CMD_COMPONENT_ARM_DISARM", 
            176: "VEHICLE_CMD_DO_SET_MODE",
        }
        self.vehicle_position = [0.0, 0.0, 0.0]
        #self.trajectory_waypoints = square_vertices(self.square_side, self.square_height)
        self.trajectory_waypoints = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        self.wayp_idx = 0

        self.logger = self.get_logger()
        self.logger.info("Global planner node initialized")

        self.timer = self.create_timer(0.1, self.waypoint_callback)

    def octomap_pc2_callback(self, pointcloud: PointCloud2) -> None:
        self.get_logger().info("Octomap pointcloud received")
        pc2_list = []
        
        for p in point_cloud2.read_points(pointcloud, field_names = ("x", "y", "z"), skip_nans=True):
            # Get XYZ coordinates to calculate vertical angle and filter by vertical scans
            x = p[0]
            y = p[1]
            z = p[2]

            pc2_list.append([x, y, z])

        if self.save_pc2_octomap: 
            with open("pc2_octomap_list", "wb") as fp:   #Pickling
                pickle.dump(pc2_list, fp)

            self.save_pc2_octomap = False
                    
        pc2_cropped = PointCloud2()
        pc2_cropped.header = pointcloud.header
        pc2_cropped.height = 1
        pc2_cropped.width = len(pc2_list)
        pc2_cropped.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        pc2_cropped.is_bigendian = False
        pc2_cropped.point_step = 12  # 4 (x) + 4 (y) + 4 (z) bytes per point
        pc2_cropped.row_step = pc2_cropped.point_step * len(pc2_list)
        pc2_cropped.data = np.array(pc2_list, dtype=np.float32).tobytes()
        self.pc2_pub.publish(pc2_cropped)


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
        
        if self.mode == 'square': 
            target_distance = self.distance_to_vertex(self.wayp_idx)
            self.get_logger().info(f"Distance to next waypoint {target_distance} m")

            if target_distance < 0.3: 
                time.sleep(10)
                if self.wayp_idx == 3: 
                    self.wayp_idx = 0
                else:
                    self.wayp_idx += 1

            target_waypoint = self.trajectory_waypoints[self.wayp_idx]
            wayp_msg = self.create_waypoint_msg(target_waypoint[0], target_waypoint[1], target_waypoint[2])
            self.waypoint_publisher.publish(wayp_msg)

    def create_waypoint_msg(self, x, y, z):
        waypoint_msg = Vector3Stamped()

        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.header.frame_id = 'base_link'

        waypoint_msg.vector.x = x
        waypoint_msg.vector.y = y
        waypoint_msg.vector.z = z 
        
        return waypoint_msg
    
    
    
    def distance_to_vertex(self, vertex_index):
        """
        Calculate the distance between the vehicle position and a square vertex.

        Args:
            vertex_index (int): Index of the vertex (0 to 3).

        Returns:
            float: The distance between the vehicle position and the specified vertex.
        """
        # Ensure valid vertex index
        if vertex_index < 0 or vertex_index > 3:
            self.logger.error("Invalid vertex index. It must be between 0 and 3.")
            return None

        # Get coordinates of the specified vertex
        vertex = self.trajectory_waypoints[vertex_index]

        # Calculate distance using Euclidean distance formula
        dx = self.vehicle_position[0] - vertex[0]
        dy = self.vehicle_position[1] - vertex[1]
        dz = self.vehicle_position[2] - vertex[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        return distance

def main(): 
    rclpy.init()

    global_planner = GlobalPlanner()

    rclpy.spin(global_planner)

    global_planner.destroy_node()
    rclpy.shutdown()