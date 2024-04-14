import rclpy 
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math

from geometry_msgs.msg import Vector3Stamped

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

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

        # Publishers
        self.waypoint_publisher = self.create_publisher(Vector3Stamped, "/global_waypoint", qos_profile)

        # Parameters
        self.square_side = 4.0
        self.square_height = 1.0

        # Initialize variables
        self.command_mapping = {
            21: "VEHICLE_CMD_NAV_LAND", 
            400: "VEHICLE_CMD_COMPONENT_ARM_DISARM", 
            176: "VEHICLE_CMD_DO_SET_MODE",
        }
        self.vehicle_position = [0.0, 0.0, 0.0]
        self.trajectory_waypoints = self.square_vertices()
        self.wayp_idx = 0

        self.logger = self.get_logger()
        self.logger.info("Global planner node initialized")

        self.timer = self.create_timer(0.1, self.waypoint_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        # Convert NED (px4) to ENU (ROS)
        x = vehicle_local_position.x
        y = -vehicle_local_position.y
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

        target_distance = self.distance_to_vertex(self.wayp_idx)
        self.get_logger().info(f"Distance to next waypoint {target_distance} m")

        if target_distance < 0.3: 
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
    
    def square_vertices(self):
        """
        Calculate the coordinates of all four vertices of a square given one vertex.
        
        Args:
            x (float): x-coordinate of one vertex.
            y (float): y-coordinate of one vertex.
            self.square_side (float): Length of one side of the square.
            height (float): Height of the square.
            
        Returns:
            tuple: A tuple containing four tuples, each representing the coordinates of one vertex.
        """
        # Calculate coordinates of other three vertices
        x1 = self.square_side
        y1 = 0.0
        x2 = self.square_side
        y2 = self.square_side
        x3 = 0.0
        y3 = self.square_side

        # Return the coordinates as a tuple of tuples
        return [[0.0, 0.0, self.square_height], [x1, y1, self.square_height], [x2, y2, self.square_height], [x3, y3, self.square_height]]
    
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