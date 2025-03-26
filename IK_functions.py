#! /usr/bin/env python3
import numpy as np
import math as m
from functools import reduce
"""
    # {Jiarun Han}
    # {jiarunh@kth.se}
"""

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]

    l0=0.07
    l1 = 0.3
    l2 = 0.35
    x0 = x - l0
    y0 = y
    z0 = z

    r = m.sqrt(x0**2 + y0**2)
    cos_q2 = (x0**2 + y0**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    if cos_q2 < -1.0:
        cos_q2 = -1.0
    elif cos_q2 > 1.0:
        cos_q2 = 1.0

    sin_q2 = m.sqrt(1 - cos_q2**2)  
    q2 = m.atan2(sin_q2, cos_q2)

    k1 = l1 + l2 * cos_q2
    k2 = l2 * sin_q2
    q1 = m.atan2(y0, x0) - m.atan2(k2, k1)
    q3 = z0  
    
    q = [q1, q2, q3]
    return q


def kuka_IK(target_position, target_rotation, current_joint_angles):
    x_target, y_target, z_target = target_position
    joint_angles = np.array(current_joint_angles)  
    error_tolerance = 1e-2  
    desired_position = target_position

    for iteration in range(100):  
        current_position, current_rotation = compute_DH_transform(joint_angles)  
        position_error = current_position - desired_position  
        rotation_error = compute_rotation_error(target_rotation, current_rotation)  
        combined_error = np.concatenate((position_error, rotation_error))

        jacobian_matrix = compute_jacobian(joint_angles) 
        joint_adjustments = np.dot(np.linalg.pinv(jacobian_matrix), combined_error)  
        joint_angles = joint_angles - joint_adjustments 

        if np.max(np.abs(combined_error)) < error_tolerance:
            break

    return joint_angles  

def compute_DH_transform(joint_angles, short_mode=False):
    link_offset_A, link_offset_B, link_offset_C, link_offset_D = 0.331, 0.4, 0.39, 0.078
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = joint_angles
    
    dh_table = [
        [m.pi/2, 0, 0, theta1],
        [-m.pi/2, 0, 0, theta2],
        [-m.pi/2, link_offset_B, 0, theta3],
        [m.pi/2, 0, 0, theta4],
        [m.pi/2, link_offset_C, 0, theta5],
        [-m.pi/2, 0, 0, theta6],
        [0, 0, 0, theta7]
    ]
    
    transformation_matrices = list(map(lambda params: generate_transformation_matrix(*params), dh_table))
    final_transformation = reduce(np.dot, transformation_matrices)
    rotation_matrix = final_transformation[:3, :3]
    position_vector = np.dot(final_transformation, np.array([0, 0, link_offset_D, 1]))[:3]
    
    if not short_mode:
        position_vector[2] += link_offset_A  

    return position_vector, rotation_matrix
def compute_rotation_error(target_rotation, current_rotation):
    target_rotation = np.array(target_rotation)
    current_rotation = np.array(current_rotation)
    
    error_vector = 0.5 * (np.cross(target_rotation[:, 0], current_rotation[:, 0]) +
                          np.cross(target_rotation[:, 1], current_rotation[:, 1]) +
                          np.cross(target_rotation[:, 2], current_rotation[:, 2]))
    
    return error_vector

def generate_transformation_matrix(alpha, d, r, theta):
    matrix = np.array([
        [m.cos(theta), -m.sin(theta)*m.cos(alpha), m.sin(theta)*m.sin(alpha), r*m.cos(theta)],
        [m.sin(theta), m.cos(theta)*m.cos(alpha), -m.cos(theta)*m.sin(alpha), r*m.sin(theta)],
        [0, m.sin(alpha), m.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return matrix
def compute_jacobian(joint_angles):
    link_offset_A, link_offset_B, link_offset_C, link_offset_D = 0.331, 0.4, 0.39, 0.078
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = joint_angles
    
    dh_table = [
        [m.pi/2, 0, 0, theta1],
        [-m.pi/2, 0, 0, theta2],
        [-m.pi/2, link_offset_B, 0, theta3],
        [m.pi/2, 0, 0, theta4],
        [m.pi/2, link_offset_C, 0, theta5],
        [-m.pi/2, 0, 0, theta6],
        [0, 0, 0, theta7]
    ]
    
    end_effector_position, _ = compute_DH_transform(joint_angles, short_mode=True)
    transformation_matrices = list(map(lambda params: generate_transformation_matrix(*params), dh_table))
    
    transforms = [reduce(np.dot, transformation_matrices[:i], np.eye(4)) for i in range(len(transformation_matrices))]
    rotation_matrices = list(map(lambda T: T[:3, :3], transforms))
    translation_vectors = list(map(lambda T: T[:3, 3], transforms))
    
    jacobian_columns = []
    for i in range(7):
        rotation = rotation_matrices[i]
        translation = translation_vectors[i]
        z_axis = np.dot(rotation, np.array([0., 0., 1.]))
        linear_velocity = np.cross(z_axis, end_effector_position - translation)
        angular_velocity = z_axis
        jacobian_column = np.concatenate((linear_velocity, angular_velocity))
        jacobian_columns.append(jacobian_column)
    
    jacobian_matrix = np.stack(jacobian_columns).T
    return jacobian_matrix

