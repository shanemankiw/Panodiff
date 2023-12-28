import numpy as np
import cv2
import math
import plotly.graph_objs as go
from PIL import ImageFile
from PIL import Image
import torch
import torch.nn.functional as F

def gen_perspective_patch(ph, pw, fov):
    '''
    fov is in angle mode.(180/360)
    return
    '''
    fov_rad = math.radians(fov)
    width_sphere_half = np.sin(fov_rad/2)
    height_sphere_half = width_sphere_half * ph / pw
    distance_to_plane = np.cos(fov_rad/2)
    yy, zz = np.meshgrid(np.linspace(-width_sphere_half, width_sphere_half, pw), \
                         np.linspace(height_sphere_half, -height_sphere_half, ph))
    
    
    x = np.ones_like(zz) * distance_to_plane
    y = yy
    z = zz
    
    patch = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    patch /= np.linalg.norm(patch, axis=2)[..., None]

    return patch

def rotate_patch_euler(theta, phi, patch):
    if phi==0 and theta ==0:
        return patch.copy()
        
    H, W, _ = patch.shape
    phi = phi / 180 * np.pi
    theta = theta / 180 * np.pi

    R_lattitude = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    patch_center = np.array([1, 0, 0])
    z_axis = np.array([0,0,1])
    new_axis = -np.cross(z_axis, np.matmul(R_lattitude, patch_center))
    
    R_longitude = np.cos(phi) * np.eye(3) + np.sin(phi) * skew(new_axis) + (1 - np.cos(phi)) * np.outer(new_axis, new_axis)
    R_final = R_longitude @ R_lattitude

    rotated_patch_flat = np.einsum('jk, hwk -> hwj', R_final, patch)
    rotated_patch = rotated_patch_flat.reshape(H, W, 3)

    # debugging
    '''x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    target_dir = np.array([x, y, z])

    data = []
    data.append(draw_line(patch[128, 128], cam_id='original'))
    data.append(draw_line(patch[60, 60], cam_id='original_60_60'))
    data.append(draw_line(new_axis, cam_id='new_axis'))
    data.append(draw_line(target_dir, cam_id='target_dir'))
    
    data.append(draw_line(rotated_patch[128, 128], cam_id='rotated_center'))
    data.append(draw_line(rotated_patch[60, 60], cam_id='rotated_60_60'))
    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()
    a = 1'''

    return rotated_patch

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def gen_patch_thetaphi(patch):
    #patch_norm = np.linalg.norm(patch, ord=2, axis=2)
    phi = -np.arcsin(patch[..., 2]) / (np.pi/2)
    theta = np.arctan2(patch[..., 1], patch[..., 0]) / (np.pi)

    # regularize
    phi[phi > 1] -= 2
    phi[phi < -1] += 2
    theta[theta > 1] -= 2
    theta[theta < -1] += 2

    return np.stack([theta, phi], axis=-1) # radians

def gen_distort_fov(pano_shape, crop_shape, theta, phi):
    # Get image dimensions
    H, W = pano_shape[0], pano_shape[1]
    crop_h, crop_w = crop_shape[0], crop_shape[1]
    
    center_x = theta/180 + 1
    center_y = 1 - phi/90
    
    left = center_x - crop_w / W
    right = center_x + crop_w / W
    bottom = center_y - crop_h / H
    top = center_y + crop_h / H
    
    xx, yy = np.meshgrid(np.linspace(left, right, crop_w), np.linspace(bottom, top, crop_h))
    xx = np.mod(xx, 2) - 1
    yy = np.mod(yy, 2) - 1
    
    sample_grid = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    return sample_grid

def gen_undistort_fov(crop_shape, theta, phi, fov=90):
    crop_h, crop_w = crop_shape[0], crop_shape[1]
    patch = gen_perspective_patch(crop_h, crop_w, fov) 
    patch_rotated = rotate_patch_euler(theta, phi, patch)
    sample_grid = gen_patch_thetaphi(patch_rotated)
    return sample_grid

def gen_sample_grid(pano_shape, crop_shape, theta, phi, fov=90, undist=False):
    if undist:
        sample_grid = gen_undistort_fov(crop_shape, theta, phi, fov)
    else:
        sample_grid = gen_distort_fov(pano_shape, crop_shape, theta, phi)
    
    return sample_grid
    
if __name__ == "__main__":
    Pano_dir = '/data/chenziyu/myprojects/PanoData/SUN360/train/010705.jpg'
    # with open(Pano_dir, 'rb') as f:
    #     img = Image.open(f)
    #     img = img.convert('RGB')
    Pano = np.asarray(cv2.imread(Pano_dir)).astype(np.float64)
    
    theta, phi = 130, -30
    crop_res= [256, 256]

    sample_grid = gen_sample_grid(Pano.shape[:2], crop_res, theta, phi)
    undist_sample_grid = gen_sample_grid(Pano.shape[:2], crop_res, theta, phi, fov=90, undist=True)
    
    
    sample_grid = torch.from_numpy(sample_grid[None, :])
    undist_sample_grid = torch.from_numpy(undist_sample_grid[None, :])
    
    Pano = torch.from_numpy(Pano[None,:]).permute([0,3,1,2])
    patch = F.grid_sample(Pano, sample_grid, align_corners=True)
    patch = patch[0].numpy().transpose([1,2,0])
    
    undist_patch = F.grid_sample(Pano, undist_sample_grid, align_corners=True)
    undist_patch = undist_patch[0].numpy().transpose([1,2,0])
    
    cat_patch = np.concatenate([patch, undist_patch], axis=1)

    cv2.imwrite('/data/chenziyu/myprojects/ExtremeRotation_code/vis_image/theta{}_phi{}.png'.format('%04d'%theta, '%04d'%phi), cat_patch)

