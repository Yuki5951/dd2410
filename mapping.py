#!/usr/bin/env python3

"""
    # {Jiarun Han}
    # {student id}
    # {jiarunh@kth.se}
"""

# Python standard library
from math import cos, sin, atan2, fabs

# Numpy
import numpy as np

# "Local version" of ROS messages
from local.geometry_msgs import PoseStamped, Quaternion
from local.sensor_msgs import LaserScan
from local.map_msgs import OccupancyGridUpdate

from grid_map import GridMap


class Mapping:
    def __init__(self, unknown_space, free_space, c_space, occupied_space,
                 radius, optional=None):
        self.unknown_space = unknown_space
        self.free_space = free_space
        self.c_space = c_space
        self.occupied_space = occupied_space
        self.allowed_values_in_map = {"self.unknown_space": self.unknown_space,
                                      "self.free_space": self.free_space,
                                      "self.c_space": self.c_space,
                                      "self.occupied_space": self.occupied_space}
        self.radius = radius
        self.__optional = optional

    def get_yaw(self, q):
        """Returns the Euler yaw from a quaternion.
        :type q: Quaternion
        """
        return atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))

    def raytrace(self, start, end):
        """Returns all cells in the grid map that has been traversed
        from start to end, including start and excluding end.
        start = (x, y) grid map index
        end = (x, y) grid map index
        """
        (start_x, start_y) = start
        (end_x, end_y) = end
        x = start_x
        y = start_y
        (dx, dy) = (fabs(end_x - start_x), fabs(end_y - start_y))
        n = dx + dy
        x_inc = 1
        if end_x <= start_x:
            x_inc = -1
        y_inc = 1
        if end_y <= start_y:
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2

        traversed = []
        for i in range(0, int(n)):
            traversed.append((int(x), int(y)))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                if error == 0:
                    traversed.append((int(x + x_inc), int(y)))
                y += y_inc
                error += dx

        return traversed

    def add_to_map(self, grid_map, x, y, value):
        """Adds value to index (x, y) in grid_map if index is in bounds.
        Returns weather (x, y) is inside grid_map or not.
        """
        if value not in self.allowed_values_in_map.values():
            raise Exception("{0} is not an allowed value to be added to the map. "
                            .format(value) + "Allowed values are: {0}. "
                            .format(self.allowed_values_in_map.keys()) +
                            "Which can be found in the '__init__' function.")

        if self.is_in_bounds(grid_map, x, y):
            grid_map[x, y] = value
            return True
        return False

    def is_in_bounds(self, grid_map, x, y):
        """Returns weather (x, y) is inside grid_map or not."""
        if x >= 0 and x < grid_map.get_width():
            if y >= 0 and y < grid_map.get_height():
                return True
        return False



    def update_map(self, grid_map, pose, scan):
        robot_yaw = self.get_yaw(pose.pose.orientation)
        origin = grid_map.get_origin()
        resolution = grid_map.get_resolution()
        
        obs = []
        x_list = []
        y_list = []
        robot_x = int((pose.pose.position.x - origin.position.x) / resolution)
        robot_y = int((pose.pose.position.y - origin.position.y) / resolution)
        
        
        for i, range_laser in enumerate(scan.ranges):
            if scan.range_min < range_laser < scan.range_max:
                angle_laser = scan.angle_min + i * scan.angle_increment
                
                x_laser = pose.pose.position.x + range_laser * cos(robot_yaw + angle_laser)
                y_laser = pose.pose.position.y + range_laser * sin(robot_yaw + angle_laser)
                
                grid_x = int((x_laser - origin.position.x) / resolution)
                grid_y = int((y_laser - origin.position.y) / resolution)
                
                # Update min and max for both free and occupied space
                x_list.append(grid_x)
                y_list.append(grid_y)

                obs.append((grid_x, grid_y))
                # Raytrace for free space
                free_cells = self.raytrace([int((pose.pose.position.x - origin.position.x) / resolution), int((pose.pose.position.y - origin.position.y) / resolution)], [grid_x, grid_y])
                for free_cell in free_cells:
                    (cell_x, cell_y) = free_cell
                    self.add_to_map(grid_map, cell_x, cell_y, self.free_space)   
                
        for obstacle in obs:
            grid_x, grid_y = obstacle
            if self.is_in_bounds(grid_map, grid_x, grid_y):
                self.add_to_map(grid_map, grid_x, grid_y, self.occupied_space)
        # Prepare the update message
        update = OccupancyGridUpdate()
        update.x = min(x_list)
        update.y = min(y_list)
        update.width = max(x_list) - min(x_list) + 1
        update.height = max(y_list) - min(y_list) + 1
        
        update.data = []
        for y in range(update.height):
            for x in range(update.width):
                update.data.append(grid_map.__getitem__([y, x]))
        
        return grid_map, update

    def inflate_map(self, grid_map):
        for y in range(grid_map.get_height()):
            for x in range(grid_map.get_width()):
                if grid_map[x, y] == self.occupied_space:
                    for dx in range(-self.radius-1, self.radius + 1):
                        for dy in range(-self.radius-1, self.radius + 1):
                            if dx*dx + dy*dy <= self.radius*self.radius:
                                nx, ny = x + dx, y + dy
                                if self.is_in_bounds(grid_map, nx, ny):
                                    if grid_map[nx, ny] != self.occupied_space:
                                        self.add_to_map(grid_map, nx, ny, self.c_space)
        
        return grid_map
