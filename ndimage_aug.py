import numpy as np
import math
import random
from scipy import ndimage

def do_random_transform(x, x_rotation_max_angel_deg, y_rotation_max_angel_deg, z_rotation_max_angel_deg):
    x_rot_mat = np.eye(3, 3)
    if x_rotation_max_angel_deg > 0:
        rot_deg = random.randint(-1 * x_rotation_max_angel_deg, x_rotation_max_angel_deg)
        x_rot_mat[1, 1] = math.cos(math.radians(rot_deg))
        x_rot_mat[2, 2] = math.cos(math.radians(rot_deg))
        x_rot_mat[1, 2] = -1 * math.sin(math.radians(rot_deg))
        x_rot_mat[2, 1] = math.sin(math.radians(rot_deg))

    y_rot_mat = np.eye(3, 3)
    if y_rotation_max_angel_deg > 0:
        rot_deg = random.randint(-1 * y_rotation_max_angel_deg, y_rotation_max_angel_deg)
        y_rot_mat[0, 0] = math.cos(math.radians(rot_deg))
        y_rot_mat[2, 2] = math.cos(math.radians(rot_deg))
        y_rot_mat[2, 0] = -1 * math.sin(math.radians(rot_deg))
        y_rot_mat[0, 2] = math.sin(math.radians(rot_deg))

    z_rot_mat = np.eye(3, 3)
    if z_rotation_max_angel_deg > 0:
        rot_deg = random.randint(-1 * z_rotation_max_angel_deg, z_rotation_max_angel_deg)
        z_rot_mat[0, 0] = math.cos(math.radians(rot_deg))
        z_rot_mat[1, 1] = math.cos(math.radians(rot_deg))
        z_rot_mat[0, 1] = -1 * math.sin(math.radians(rot_deg))
        z_rot_mat[1, 0] = math.sin(math.radians(rot_deg))

    full_rot_mat = np.dot(np.dot(x_rot_mat, y_rot_mat), z_rot_mat)
    x = ndimage.affine_transform(x, full_rot_mat)
    return x