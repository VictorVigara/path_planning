import math
import pickle
import time
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path
from px4_msgs.msg import VehicleCommand, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray

from .trajectory_generation.rrt_algorithms.rrt.rrt import RRT
from .trajectory_generation.rrt_algorithms.rrt.rrt_star import RRTStar
from .trajectory_generation.rrt_algorithms.search_space.search_space import SearchSpace
from .trajectory_generation.rrt_algorithms.utilities.geometry import steer


class VehicleCommandMapping:
    VEHICLE_CMD_NAV_LAND = 21
    VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
    VEHICLE_CMD_DO_SET_MODE = 176


class RRT_Mode:
    RRT = "rrt"
    RRT_STAR = "rrt_star"


class GlobalPlanner(Node):
    def __init__(self) -> None:
        super().__init__("global_planner")

        ### Parameters ###########################################
        self.mode = RRT_Mode.RRT_STAR

        # Use octomap
        self.use_octomap = False

        # Octomap
        self.octomap_resolution = 0.7  # Octomap resolution is 0.1, but when inserted in search space with the same
        # resolution, there could be small spaces that could led to paths between the obstacle.
        # So, the seacrh space is set up with a bit of lower resolution to fill the gaps.

        # RRT
        self.RRT_search_space_range_x = (-1, 5)
        self.RRT_search_space_range_y = (-4, 4)
        self.RRT_search_space_range_z = (1, 1.5)
        self.RRT_goal = (4, -2, 1.5)
        self.RRT_initial = (0, 0, 1)
        self.RRT_q = 0.3  # length of tree edges
        self.RRT_r = (
            0.1  # length of smallest edge to check for intersection with obstacles
        )
        self.RRT_max_samples = 3000  # max number of samples to take before timing out
        self.RRT_prc = 0.1  # probability of checking for a connection to goal
        # RRT*
        self.RRT_rewire_count = 32  # optional, number of nearby branches to rewire

        self.save_pc2_octomap = True

        # Octomap pointcloud box filter
        self.x_crop = 0.44
        self.y_crop = 0.44

        ###########################################################

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        ### Subscribers ###
        # self.vehicle_local_position_subscriber = self.create_subscription(
        #     VehicleLocalPosition,
        #     "/fmu/out/vehicle_local_position",
        #     self.vehicle_local_position_callback,
        #     qos_profile,
        # )

        self.ekf2_px4_pos_sub = self.create_subscription(
            PoseStamped, 
            '/px4_ekf_pose', 
            self.ekf_px4_pose_callback, 
            1
        )

        self.vehicle_command_subscriber = self.create_subscription(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            self.vehicle_command_callback,
            qos_profile,
        )

        self.octomap_pc2_sub = self.create_subscription(
            PointCloud2, "/octomap_point_cloud_centers", self.octomap_pc2_callback, 1
        )

        self.contact_detection_sub = self.create_subscription(
            Float32MultiArray, "/collision_detection", self.contact_callback, 1
        )

        ### Publishers ###
        self.waypoint_publisher = self.create_publisher(
            Vector3Stamped, "/global_waypoint", qos_profile
        )

        self.pc2_pub = self.create_publisher(PointCloud2, "read_pc", 1)

        self.vehicle_path_pub = self.create_publisher(Path, "/RRT_path", 1)
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
        self.trajectory_waypoints = []
        self.wayp_idx = 0
        self.goal_reached = False
        self.initial_RRT_solved = False
        self.octomap_collision = False
        # Platform collision variables
        self.platform_collision = False  # Set to True when collision detected
        self.collision_recovering = (
            False  # set to True when previous waypoint sent and not reached
        )
        # Once previous waypoint has been reached, it will be set to False, so
        # If still collision, go ot the previous waypoint again

        self.get_logger().info("Initializing RRT search space")
        self.X_dimensions = np.array(
            [
                self.RRT_search_space_range_x,
                self.RRT_search_space_range_y,
                self.RRT_search_space_range_z,
            ]
        )
        # Create search space
        self.X = SearchSpace(self.X_dimensions)

        self.logger = self.get_logger()
        self.logger.info("Global planner node initialized")

    def ekf_px4_pose_callback(self, ekf_pose_msg: PoseStamped) -> None: 
        #print("Receiving px4 ekf pose")
        x = ekf_pose_msg.pose.position.x
        y = ekf_pose_msg.pose.position.y
        z = ekf_pose_msg.pose.position.z
        self.vehicle_position = [x, y,z]

    def contact_callback(self, contact_msg: Float32MultiArray) -> None:
        collision = contact_msg.data[0]
        if collision == 0.0:
            self.platform_collision = False
        elif collision == 1.0:
            self.platform_collision = True

        self.platform_collision_orientation = contact_msg.data[1]
        self.platform_collision_displacement = contact_msg.data[2]

        if self.platform_collision and self.collision_recovering == False:
            self.get_logger().info(
                f"Receiving collision from {self.platform_collision_orientation} - {self.platform_collision_displacement} cm"
            )

    def octomap_pc2_callback(self, pointcloud: PointCloud2) -> None:
        # self.get_logger().info("Octomap pointcloud received")
        # self.obstacles = []
        self.octomap_occupied_pointcloud = []
        pc_init_time = time.time()
        self.X = SearchSpace(self.X_dimensions)
        init_ss_time = time.time()
        #self.get_logger().info(f"Init search space: {(init_ss_time-pc_init_time)*1000}")
        for p in point_cloud2.read_points(
            pointcloud, field_names=("x", "y", "z"), skip_nans=True
        ):
            # Get XYZ coordinates to calculate vertical angle and filter by vertical scans
            x = p[0]
            y = p[1]
            z = p[2]

            if z > 0.1 and (
                (x > self.x_crop or x < -self.x_crop)
                or (y > self.y_crop or y < -self.y_crop)
            ):
                obstacle = self.point_to_obstacle([x, y, z])
                self.X.obs.insert(uuid.uuid4().int, tuple(obstacle), tuple(obstacle))
                # self.obstacles.append(obstacle)
                """ self.octomap_occupied_pointcloud.append([x, y, z]) """
        pc_final_time = time.time()
        #self.get_logger().info(f"Fill search space: {(pc_final_time-init_ss_time)*1000}")

        # self.get_logger().info(f"Ptcloud callback time: {(pc_final_time-pc_init_time)*1000}")
        """ if self.save_pc2_octomap:
            with open("pc2_octomap_list", "wb") as fp:  # Pickling
                pickle.dump(self.octomap_occupied_pointcloud, fp)

            self.save_pc2_octomap = False

        pc2_cropped = PointCloud2()
        pc2_cropped.header = pointcloud.header
        pc2_cropped.height = 1
        pc2_cropped.width = len(self.octomap_occupied_pointcloud)
        pc2_cropped.fields.append(
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.fields.append(
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.fields.append(
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.is_bigendian = False
        pc2_cropped.point_step = 12  # 4 (x) + 4 (y) + 4 (z) bytes per point
        pc2_cropped.row_step = pc2_cropped.point_step * len(
            self.octomap_occupied_pointcloud
        )
        pc2_cropped.data = np.array(
            self.octomap_occupied_pointcloud, dtype=np.float32
        ).tobytes()
        self.pc2_pub.publish(pc2_cropped) """

        self.octomap_received = True

    # def vehicle_local_position_callback(self, vehicle_local_position):
    #     """Callback function for vehicle_local_position topic subscriber."""
    #     # Convert NED (px4) to ENU (ROS)
    #     x = vehicle_local_position.y
    #     y = vehicle_local_position.x
    #     z = -vehicle_local_position.z

    #     self.vehicle_position = [x, y, z]

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

        if not self.use_octomap: 
            if not self.initial_RRT_solved:
                self.get_logger().info("Solving 1st RRT ...")
                self.trajectory_waypoints = self.solve_RRT(
                    initial_waypoint=self.RRT_initial
                )
                if self.trajectory_waypoints == None:
                    self.get_logger().info(
                        f"Initial Path RRT no solution, will try again"
                    )
                    self.initial_RRT_solved = False

                else:
                    self.get_logger().info(
                        f"Initial Path RRT: {self.trajectory_waypoints}"
                    )
                    self.initial_RRT_solved = True

        if self.octomap_received and self.use_octomap:
            # Calculates initial RRT
            if not self.initial_RRT_solved:
                self.get_logger().info("Solving 1st RRT ...")
                self.trajectory_waypoints = self.solve_RRT(
                    initial_waypoint=self.RRT_initial
                )
                if self.trajectory_waypoints == None:
                    self.get_logger().info(
                        f"Initial Path RRT no solution, will try again"
                    )
                    self.initial_RRT_solved = False

                else:
                    self.get_logger().info(
                        f"Initial Path RRT: {self.trajectory_waypoints}"
                    )
                    self.initial_RRT_solved = True
            else:
                # self.get_logger().info("Checking for collisions ...")
                # Check previous RRT solution new octomap collisions
                # TODO: Check for collision with the new map update
                # self.octomap_collision = True
                time_check_i = time.time()
                previous_wayp = self.trajectory_waypoints[
                    : self.wayp_idx
                ].copy()  # Waypoints already done
                wayp_to_check = self.trajectory_waypoints[
                    self.wayp_idx :
                ].copy()  # Following waypoints to check a possible collision
                for idx in range(len(wayp_to_check) - 1):
                    if not self.X.collision_free(
                        wayp_to_check[idx], wayp_to_check[idx + 1], self.RRT_r
                    ):
                        self.get_logger().info(f"Collision detected in wayp {idx}")
                        self.octomap_collision = True
                        break
                time_check_f = time.time()

                #print(f"Time to check collision: {(time_check_f-time_check_i)*1000}")
                if self.octomap_collision:

                    # TODO: Recalculate a path between the previous waypoint without collision and the goal
                    initial_recalculated = (
                        self.trajectory_waypoints[self.wayp_idx][0],
                        self.trajectory_waypoints[self.wayp_idx][1],
                        self.trajectory_waypoints[self.wayp_idx][2],
                    )
                    self.get_logger().info(f"RRT_initial: {initial_recalculated}")
                    print(self.RRT_initial)
                    recalculated_path = self.solve_RRT(
                        initial_waypoint=initial_recalculated
                    )

                    if recalculated_path == None:
                        self.get_logger().info(
                            f"Recalculated Path RRT no solution, will try again"
                        )

                    else:
                        self.get_logger().info(f"Path already done: {previous_wayp}")
                        self.get_logger().info(
                            f"Path recalculated: {recalculated_path}"
                        )
                        self.trajectory_waypoints = recalculated_path
                        self.wayp_idx = 0
                        self.get_logger().info(
                            f"Global path updated: {self.trajectory_waypoints}"
                        )
                        self.octomap_collision = False

            self.octomap_received = False

        if self.platform_collision:
            ## TODO: Create a new platform collision object space to sum it up with the octomap space
            ## TODO: Logic to handle platform collision and go one waypoint back and include
            ##      the collision in the map.
            pass

        if self.initial_RRT_solved and self.goal_reached == False:

            # If collision detected, go to the  previous waypoint
            if self.platform_collision and self.collision_recovering == False:
                if self.wayp_idx != 0:
                    self.wayp_idx -= 1
                    # REcovery flag to not go to another previous waypoint while recovering
                    self.collision_recovering = True
                    print(f"Collision previous waypoint {self.wayp_idx}")
            target_distance = self.distance_to_target(self.wayp_idx)
            

            if target_distance < 0.1:
                if self.wayp_idx == (self.n_waypoints - 1):
                    pass
                    # self.wayp_idx = 0
                    # self.goal_reached = True
                elif self.octomap_collision == False and self.platform_collision == False:
                    self.wayp_idx += 1
                    print(f"Next waypoint {self.wayp_idx}")
                    # Recover collision when previous target waypoint reached
                    self.collision_recovering = False
                elif self.platform_collision == True and self.collision_recovering == True: 
                    # Set to false so next iteration we will go to previous waypoint
                    self.collision_recovering = False

            
            #self.get_logger().info(f"Current target waypoint {self.wayp_idx}")
            print(f"Curr wayp: {self.wayp_idx} - dist: {target_distance}")

            target_waypoint = self.trajectory_waypoints[self.wayp_idx]
            wayp_msg = self.create_waypoint_msg(
                target_waypoint[0], target_waypoint[1], target_waypoint[2]
            )
            self.waypoint_publisher.publish(wayp_msg)

    def solve_RRT(self, initial_waypoint):
        traj_wayp = []
        initial_rrt_time = time.time()

        if self.mode == RRT_Mode.RRT:
            self.get_logger().info("Solving RRT")
            rrt = RRT(
                self.X,
                self.RRT_q,
                initial_waypoint,
                self.RRT_goal,
                self.RRT_max_samples,
                self.RRT_r,
                self.RRT_prc,
            )
        elif self.mode == RRT_Mode.RRT_STAR:
            self.get_logger().info("Solving RRT STAR")
            rrt = RRTStar(
                self.X,
                self.RRT_q,
                initial_waypoint,
                self.RRT_goal,
                self.RRT_max_samples,
                self.RRT_r,
                self.RRT_prc,
                self.RRT_rewire_count,
            )
        path = rrt.rrt_star()

        # Exit if no path foun
        if path == None:
            return None

        final_time = time.time()
        self.get_logger().info(f"RRT solved in {(final_time-initial_rrt_time)*1000} ms")
        self.get_logger().info(f"Path RRT: {path} ")

        self.vehicle_path_msg.header.frame_id = "world"
        self.vehicle_path_msg.header.stamp = self.get_clock().now().to_msg()

        self.n_waypoints = len(path)
        # self.get_logger().info(f"Initial RRT wayps: {self.n_waypoints}")

        # Save until the penultimate waypoint in trajectory_waypoints
        for idx, waypoint in enumerate(path):
            if idx < (self.n_waypoints - 1):
                pose_msg = self.vector2PoseMsg("world", waypoint)
                self.vehicle_path_msg.poses.append(pose_msg)
                traj_wayp.append(waypoint)
                dist_next = np.sqrt(
                    (path[idx][0] - path[idx + 1][0]) ** 2
                    + (path[idx][1] - path[idx + 1][1]) ** 2
                    + (path[idx][2] - path[idx + 1][2]) ** 2
                )
                self.get_logger().info(f"Dist next waypoint: {dist_next}")
                # Check if distance with next waypoint is less than RRT_q, otherwise
                # steer waypoints between both until last distance is <= RRT_q
                if dist_next > self.RRT_q:
                    self.get_logger().info(f"Steering is needed")
                    while dist_next > self.RRT_q:
                        # Steer closest waypoint
                        steered_wayp = steer(waypoint, path[idx + 1], self.RRT_q)
                        dist_next = dist_next - self.RRT_q
                        self.get_logger().info(f"Dist next waypoint: {dist_next}")
                        traj_wayp.append(steered_wayp)

                        pose_msg = self.vector2PoseMsg("world", steered_wayp)
                        self.vehicle_path_msg.poses.append(pose_msg)
                        # Update waypoint to continue steering
                        waypoint = steered_wayp

                self.get_logger().info(
                    f"Subpath steered, continuing with next waypoint"
                )

        # Append last waypoint to the trajectory
        traj_wayp.append(path[-1])
        pose_msg = self.vector2PoseMsg("world", path[-1])
        self.vehicle_path_msg.poses.append(pose_msg)

        self.get_logger().info(f"Final path steered: {traj_wayp}")

        # save  initial solution, because trajectory waypoints will change with replanning
        self.global_trajectory_waypoints = traj_wayp.copy()
        self.n_waypoints = len(traj_wayp)
        # self.get_logger().info(f"Steered Path RRT: {traj_wayp}")

        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        return traj_wayp

    def create_waypoint_msg(self, x, y, z):
        waypoint_msg = Vector3Stamped()

        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.header.frame_id = "world"

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

        # print(f"Vehicle pose: {self.vehicle_position}")
        # print(f"Target pose: {target_waypoint}")

        # Calculate distance using Euclidean distance formula
        dx = self.vehicle_position[0] - target_waypoint[0]
        dy = self.vehicle_position[1] - target_waypoint[1]
        dz = self.vehicle_position[2] - target_waypoint[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        return distance

    def point_to_obstacle(self, point):
        x1, y1, z1 = [coord - self.octomap_resolution / 2 for coord in point]
        x2, y2, z2 = [coord + self.octomap_resolution / 2 for coord in point]
        obstacle = [x1, y1, z1, x2, y2, z2]
        return obstacle

    def vector2PoseMsg(self, frame_id, position):
        pose_msg = PoseStamped()
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        pose_msg.header.frame_id = frame_id
        """ pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3] """
        # self.get_logger().info(f"Wayp: {position}")
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


if __name__ == "__main__":
    main()
