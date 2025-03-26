#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# {Jiarun Han}
# {jiarunh@kth.se}

from dubins import *
import math
import heapq

DELTA_T = 0.01
GRID_SIZE = 0.4
ANGLE_DIVISIONS = 10
phi = 0.2

class PathSegment:
    def __init__(self, x, y, theta, cumulative_cost=0, predecessor=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.cumulative_cost = cumulative_cost
        self.predecessor = predecessor
        self.estimated_total_cost = 0
        self.duration = 0
        self.control = 0

    def __lt__(self, other):
        return self.estimated_total_cost < other.estimated_total_cost

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def discretize_state(segment):
    x_disc = int(segment.x / GRID_SIZE)
    y_disc = int(segment.y / GRID_SIZE)
    theta_disc = int(segment.theta / (2 * math.pi / ANGLE_DIVISIONS))
    return (x_disc, y_disc, theta_disc)

def is_safe(car, x, y):
    if not (car.xlb <= x <= car.xub and car.ylb <= y <= car.yub):
        return False
    for obs in car.obs:
        if euclidean_distance(x, y, obs[0], obs[1]) < (obs[2] + phi):
            return False
    return True

def generate_successors(car, current, step_duration=50):
    successors = []
    controls = [-math.pi/4, 0, math.pi/4]
    for control in controls:
        x, y, theta = current.x, current.y, current.theta
        for _ in range(step_duration):
            x, y, theta = step(car, x, y, theta, control)
            if not is_safe(car, x, y):
                break
        else:
            successor = PathSegment(x, y, theta, current.cumulative_cost + step_duration * DELTA_T, current)
            successor.duration = current.duration + step_duration
            successor.control = control
            successors.append(successor)
    return successors

def estimate_cost_to_goal(segment, goal_x, goal_y):
    return euclidean_distance(segment.x, segment.y, goal_x, goal_y)

def search_path(car):
    start = PathSegment(car.x0, car.y0, 0)
    goal_x, goal_y = car.xt, car.yt

    open_list = [start]
    closed_set = set()

    while open_list:
        current = heapq.heappop(open_list)

        if euclidean_distance(current.x, current.y, goal_x, goal_y) <= 1.5:
            return current

        current_disc = discretize_state(current)
        if current_disc in closed_set:
            continue

        closed_set.add(current_disc)

        for successor in generate_successors(car, current):
            successor.estimated_total_cost = successor.cumulative_cost + estimate_cost_to_goal(successor, goal_x, goal_y)
            heapq.heappush(open_list, successor)

    return None

def reconstruct_path(final_segment):
    path = []
    current = final_segment
    while current:
        path.append((current.duration * DELTA_T, current.control))
        current = current.predecessor
    return path[::-1]

def solution(car):
    final_segment = search_path(car)
    
    if not final_segment:
        return [], [0]
    
    path = reconstruct_path(final_segment)
    times, controls = zip(*path)
    
    return list(controls[1:]), list(times)
