import rclpy 
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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

        self.vehicle_command_subscriber = self.create_subscription(VehicleCommand, "/fmu/in/vehicle_command", self.vehicle_command_callback, qos_profile)

        self.waypoint_publisher = self.create_publisher(Vector3Stamped, "/global_waypoint", qos_profile)

        self.command_mapping = {
            21: "VEHICLE_CMD_NAV_LAND", 
            400: "VEHICLE_CMD_COMPONENT_ARM_DISARM", 
            176: "VEHICLE_CMD_DO_SET_MODE",
        }

        self.logger = self.get_logger()
        self.logger.info("Global planner node initialized")

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
                


        

        

def main(): 
    rclpy.init()

    global_planner = GlobalPlanner()

    rclpy.spin(global_planner)

    global_planner.destroy_node()
    rclpy.shutdown()