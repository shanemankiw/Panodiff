import numpy as np
import cv2
import math
import plotly.graph_objs as go
import os
import argparse
import torch
import torch.nn.functional as F

# vizualization tools
def draw_line(ray_dir, cam_id, size=2):
	
	return go.Scatter3d(
	x=[0, ray_dir[0]],
	y=[0, ray_dir[1]],
	z=[0, ray_dir[2]],
	text=['{}_start'.format(cam_id),'{}_end'.format(cam_id) ],
	hovertext=['{}_start'.format(cam_id),'{}_end'.format(cam_id) ],
	mode='lines+markers',
	marker=dict(size=size),
  )

# geometric functions
def gen_patch_fov_only(ph, pw, fov):
    '''
    fov is in angle mode.(180/360)
    return
    '''
    aspect_ratio = pw / ph
    fov_rad = math.radians(fov)
    half_fov = fov_rad / 2
    half_theta = half_fov / (2 * math.pi)
    half_phi = half_fov / (math.pi * aspect_ratio)

    yy, xx = torch.meshgrid(torch.linspace(half_phi, -half_phi, ph),torch.linspace(-half_theta, half_theta, pw))
    sin_theta = torch.sin(xx * 2 * torch.pi)
    cos_theta = torch.cos(xx * 2 * torch.pi)
    sin_phi = torch.sin(yy * torch.pi)
    cos_phi = torch.cos(yy * torch.pi)
    
    x = cos_phi * cos_theta
    y = cos_phi * sin_theta
    z = sin_phi
    
    patch = torch.cat([x[None, ...], y[None, ...], z[None, ...]], dim=0)

    return patch

def gen_perspective_patch(ph, pw, fov):
    '''
    fov is in angle mode.(180/360)
    return
    '''
    fov_rad = math.radians(fov)
    width_sphere_half = torch.sin(fov_rad/2)
    height_sphere_half = width_sphere_half * ph / pw
    distance_to_plane = torch.cos(fov_rad/2)
    yy, zz = torch.meshgrid(torch.linspace(-width_sphere_half, width_sphere_half, pw), \
                         torch.linspace(height_sphere_half, -height_sphere_half, ph))
    
    
    x = torch.ones_like(zz) * distance_to_plane
    y = yy
    z = zz
    
    patch = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    patch /= torch.linalg.norm(patch, dim=2, keepdim=True)

    return patch

def skew(v, device):
    return torch.tensor([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], device=device)

def skew_torch(v):
    # assume v is B x 3
    skew_mat = torch.zeros([v.shape[0], 3, 3]).to(v.device)
    skew_mat[:, 0, 1] = -v[:, 2]
    skew_mat[:, 1, 0] = v[:, 2]

    skew_mat[:, 0, 2] = v[:, 1]
    skew_mat[:, 2, 0] = -v[:, 1]

    skew_mat[:, 1, 2] = -v[:, 0]
    skew_mat[:, 2, 1] = v[:, 0]

    return skew_mat

def rotate_patch(theta, phi, patch):
    if phi==0 and theta ==0:
        return patch.copy()
        
    H, W, _ = patch.shape
    phi = phi / 180 * torch.pi
    theta = theta / 180 * torch.pi

    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    target_dir = torch.tensor([x, y, z])
    
    # Calculate the rotation axis and angle to rotate the patch
    patch_center = torch.tensor([1, 0, 0])  # Center of the patch is facing forward
    axis = torch.linalg.cross(patch_center, target_dir)
    axis = axis / torch.linalg.norm(axis)
    angle = -torch.arccos(torch.dot(patch_center, target_dir))

    # Rotate the patch using Rodrigues' rotation formula
    patch_flat = patch.reshape(-1, 3)
    rotated_patch_flat = patch_flat.dot(torch.cos(angle) * torch.eye(3) + torch.sin(angle) * skew(axis) + (1 - torch.cos(angle)) * torch.outer(axis, axis))
    rotated_patch = rotated_patch_flat.reshape(H, W, 3)

    # debugging
    '''data = []
    data.append(draw_line(patch_center, cam_id='original'))
    data.append(draw_line(patch[60, 60], cam_id='original_60_60'))
    data.append(draw_line(target_dir, cam_id='target'))
    data.append(draw_line(axis, cam_id='axis'))
    
    data.append(draw_line(rotated_patch[128, 128], cam_id='rotated_center'))
    data.append(draw_line(rotated_patch[60, 60], cam_id='rotated_60_60'))
    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()
    a = 1'''

    return rotated_patch

def compute_rotation_matrix_from_viewpoint(rotation_x, rotation_y, batch):
    rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
    rotay = - rotation_y.view(batch, 1).type(torch.FloatTensor)
    # rotaz = torch.zeros(batch, 1)

    c1 = torch.cos(rotax).view(batch, 1)  # batch*1
    s1 = torch.sin(rotax).view(batch, 1)  # batch*1
    c2 = torch.cos(rotay).view(batch, 1)  # batch*1
    s2 = torch.sin(rotay).view(batch, 1)  # batch*1

    # pitch --> yaw
    row1 = torch.cat((c2, s1 * s2, c1 * s2), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((torch.autograd.Variable(torch.zeros(s2.size())), c1, -s1), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((-s2, s1 * c2, c1 * c2), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix

def rotate_patch_euler_inv(theta, phi, patch):
   
    B, C, H, W = patch.shape
    device = patch.device
    phi, theta = torch.deg2rad(phi)[:, None], torch.deg2rad(theta)[:, None]
    #phi = phi / 180 * torch.pi
    #theta = theta / 180 * torch.pi

    # DEBUG
    R_lattitude = torch.eye(3)[None,:].repeat([B, 1, 1]).to(device)
    R_lattitude[:, :2, :2] = torch.stack([
        torch.cat([torch.cos(theta), -torch.sin(theta)], dim=-1),
        torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1),
        ], dim=2)

    patch_center = torch.tensor([1., 0., 0.], device=theta.device)
    z_axis = torch.tensor([0., 0., 1.], device=theta.device)[None,:].repeat([B, 1])
    new_axis = -torch.linalg.cross(z_axis, torch.matmul(R_lattitude, patch_center))
    
    R_longitude = torch.cos(phi).reshape([-1, 1, 1]) * torch.eye(3)[None,:].repeat([B, 1, 1]).to(device) + \
        torch.sin(phi).reshape([-1, 1, 1]) * skew_torch(new_axis) + \
        (1 - torch.cos(phi)).reshape([-1, 1, 1]) * (new_axis.unsqueeze(-1) * new_axis.unsqueeze(-2))
    #Rotmat = R_longitude @ R_lattitude
    
    # inverse
    R_final = torch.linalg.inv(R_longitude @ R_lattitude)

    rotated_patch = torch.einsum('bjk, bkhw -> bjhw', R_final, patch)

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

def rotate_patch_euler(theta, phi, patch):
   
    B, C, H, W = patch.shape
    device = patch.device
    phi, theta = torch.deg2rad(phi)[:, None], torch.deg2rad(theta)[:, None]
    #phi = phi / 180 * torch.pi
    #theta = theta / 180 * torch.pi

    # DEBUG
    R_lattitude = torch.eye(3)[None,:].repeat([B, 1, 1]).to(device)
    R_lattitude[:, :2, :2] = torch.stack([
        torch.cat([torch.cos(theta), -torch.sin(theta)], dim=-1),
        torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1),
        ], dim=2)

    patch_center = torch.tensor([1., 0., 0.], device=theta.device)
    z_axis = torch.tensor([0., 0., 1.], device=theta.device)[None,:].repeat([B, 1])
    new_axis = -torch.linalg.cross(z_axis, torch.matmul(R_lattitude, patch_center))
    
    R_longitude = torch.cos(phi).reshape([-1, 1, 1]) * torch.eye(3)[None,:].repeat([B, 1, 1]).to(device) + \
        torch.sin(phi).reshape([-1, 1, 1]) * skew_torch(new_axis) + \
        (1 - torch.cos(phi)).reshape([-1, 1, 1]) * (new_axis.unsqueeze(-1) * new_axis.unsqueeze(-2))
    #Rotmat = R_longitude @ R_lattitude
    
    # inverse
    R_final = R_longitude @ R_lattitude

    rotated_patch = torch.einsum('bjk, bkhw -> bjhw', R_final, patch)

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

def rotate_patch_inv(theta, phi, patch):
    if phi==0 and theta ==0:
        return patch.copy()
        
    H, W, _ = patch.shape
    phi = phi / 180 * torch.pi
    theta = theta / 180 * torch.pi

    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    target_dir = torch.tensor([x, y, z])
    
    # Calculate the rotation axis and angle to rotate the patch
    patch_center = torch.tensor([1, 0, 0])  # Center of the patch is facing forward
    axis = torch.linalg.cross(patch_center, target_dir)
    axis = axis / torch.linalg.norm(axis)
    angle = -torch.arccos(torch.dot(patch_center, target_dir))

    # Rotate the patch using Rodrigues' rotation formula
    patch_flat = patch.reshape(-1, 3)
    rot_mat = torch.cos(angle) * torch.eye(3) + torch.sin(angle) * skew(axis) + (1 - torch.cos(angle)) * torch.outer(axis, axis)
    rotated_patch_flat = patch_flat.dot(torch.linalg.inv(rot_mat))
    rotated_patch = rotated_patch_flat.reshape(H, W, 3)

    # debugging
    '''data = []
    data.append(draw_line(patch_center, cam_id='original'))
    data.append(draw_line(patch[60, 60], cam_id='original_60_60'))
    data.append(draw_line(target_dir, cam_id='target'))
    data.append(draw_line(axis, cam_id='axis'))
    
    data.append(draw_line(rotated_patch[128, 128], cam_id='rotated_center'))
    data.append(draw_line(rotated_patch[60, 60], cam_id='rotated_60_60'))
    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()
    a = 1'''

    return rotated_patch

def gen_patch_thetaphi(patch):
    #patch_norm = torch.linalg.norm(patch, ord=2, axis=2)
    phi = -torch.arcsin(patch[:, 2]) / (torch.pi/2)
    theta = torch.arctan2(patch[:, 1], patch[:, 0]) / (torch.pi)

    # regularize
    phi[phi > 1] -= 2
    phi[phi < -1] += 2
    theta[theta > 1] -= 2
    theta[theta < -1] += 2

    return torch.stack([theta, phi], axis=-1) # radians

def sample_patch_grid(Pano_warped, fov, ph, pw):
    fov_rad = torch.tensor(fov * torch.pi / 180.0, device=Pano_warped.device)
    #fov_rad = torch.tensor(fov_rad)
    width_sphere_half = torch.sin(fov_rad/2)
    height_sphere_half = width_sphere_half * ph / pw
    distance_to_plane = torch.cos(fov_rad/2)

    # u -> y -> horizontal, v -> z -> vertical
    uv_on_plane = Pano_warped * distance_to_plane / Pano_warped[:,0:1]
    # debug to see
    '''data = []
    data.append(draw_line(Pano_warped[256, 512], cam_id='center'))
    data.append(draw_line(Pano_warped[316, 45], cam_id='side'))
    data.append(draw_line(Pano_warped[496, 74], cam_id='corner'))
    
    data.append(draw_line(uv_on_plane[256, 512], cam_id='center_on_plane'))
    data.append(draw_line(uv_on_plane[316, 45], cam_id='side_on_plane'))
    data.append(draw_line(uv_on_plane[496, 74], cam_id='corner_on_plane'))
    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()
    a = 1'''
    

    mask = (uv_on_plane[:,1] < width_sphere_half) * \
            (uv_on_plane[:,1] > -width_sphere_half) * \
            (uv_on_plane[:,2] < height_sphere_half) * \
            (uv_on_plane[:,2] > -height_sphere_half) * \
            (Pano_warped[:,0] > 0)

    grid = torch.stack([uv_on_plane[:,1] / width_sphere_half, 
                     -uv_on_plane[:,2] / height_sphere_half], axis=3)
    
    return grid, mask

def sample_from_patch(patch, theta, phi, pano_height, pano_width, fov=90):
    # Pano = torch.zeros([pano_height, pano_width, 3])
    # generate Pano ray directions
    bs, _, patch_height, patch_width = patch.shape
    Pano_rays = gen_patch_fov_only(pano_height, pano_width, 360).repeat((bs, 1, 1, 1)).to(patch.device)

    # rotate the Pano_rays, such that the patch center pixel is oriented as (theta, phi)
    Pano_warped = rotate_patch_euler_inv(theta, phi, Pano_rays)

    # generate sample grid, mask
    sample_grid, mask = sample_patch_grid(Pano_warped, fov, patch_height, patch_width)
    #sample_grid[mask==0] = 0.0 # sample_grid is of shape B x H x W x 2

    sampled_Pano = F.grid_sample(patch, sample_grid, align_corners=True).permute([0, 2, 3, 1])
    sampled_Pano[mask==0] = 0

    return sampled_Pano, mask

def deform_a_little(img, delta_theta, delta_phi, pano_height, pano_width):
    '''
    Assume img is of shape BxCxHxW
    '''
    bs = img.shape[0] # assume 
    device = img.device
    #theta, phi = delta_theta.repeat([bs]), delta_phi.repeat([bs])

    Pano_rays = gen_patch_fov_only(pano_height, pano_width, 360).repeat((bs, 1, 1, 1)).to(device)
    Pano_warped = rotate_patch_euler(delta_theta, delta_phi, Pano_rays)
    #phi_grid = -torch.arcsin(Pano_warped[...,2]) / (torch.pi/2)
    #theta_grid = torch.arctan2(Pano_warped[...,1], Pano_warped[..., 0])/ (torch.pi)
    sample_grid = gen_patch_thetaphi(Pano_warped)

    return F.grid_sample(img, sample_grid, align_corners=True)

def sample_mask(theta, phi, pano_height, pano_width, fov=90):
    """
    mask.shape (pano_height, pano_width, 1)
    mask.dtype dtype('bool')
    """
    theta = torch.tensor([-theta / torch.pi * 180], dtype=torch.float32)
    phi = torch.tensor([phi / torch.pi * 180], dtype=torch.float32)
    
    Pano_rays = gen_patch_fov_only(pano_height, pano_width, 360).repeat((1, 1, 1, 1))
    
    # rotate the Pano_rays, such that the patch center pixel is oriented as (theta, phi)
    Pano_warped = rotate_patch_euler_inv(theta, phi, Pano_rays)

    # generate sample grid, mask
    _, mask = sample_patch_grid(Pano_warped, fov, 256, 256)
    
    mask = mask.permute(1, 2, 0).numpy()
    return mask

if __name__ == "__main__":
    # TODO(wjh)
    # When fov > 180, the generation would fail because the plane_distance is invalid.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Patch_dir",
        type=str,
        default="outputs/sampled_patch_theta0025_phi0045_fov120.png",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs",
    )

    parser.add_argument(
        "--pano_height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--pano_width",
        type=int,
        default=1024,
    )

    args = parser.parse_args()
    Patch = np.asarray(cv2.imread(args.Patch_dir)).astype(np.float32)
    theta = int(args.Patch_dir[args.Patch_dir.find('theta')+5:args.Patch_dir.find('theta')+9])
    phi = int(args.Patch_dir[args.Patch_dir.find('phi')+3:args.Patch_dir.find('phi')+7])
    fov = int(args.Patch_dir[-7:-4])
    ph, pw = Patch.shape[:2]
    theta = torch.tensor(theta)
    phi = torch.tensor(phi)

    # generate a hollow Panoramic Image
    Pano = torch.zeros([args.pano_height, args.pano_width, 3], dtype = torch.float32)

    # generate Pano ray directions
    Pano_rays = gen_patch_fov_only(args.pano_height, args.pano_width, 360)

    # rotate the Pano_rays, such that the patch center pixel is oriented as (theta, phi)
    Pano_warped = rotate_patch_euler_inv(theta, phi, Pano_rays)

    # generate sample grid, mask
    sample_grid, mask = sample_patch_grid(Pano_warped, fov, ph, pw)
    sample_grid[mask==0] = 0.0 # remove unavailable pixels

    import torch
    import torch.nn.functional as F

    # sample Pano, using grid
    Patch = torch.from_numpy(Patch[None,:]).permute([0,3,1,2])
    sample_grid = sample_grid[None, :]

    sampled_Pano = F.grid_sample(Patch, sample_grid, align_corners=True)
    sampled_Pano = sampled_Pano[0].numpy().transpose([1,2,0])
    sampled_Pano[mask==0] = 0
    print(sampled_Pano.shape)
    
    sampled_Pano = sampled_Pano.transpose([1, 0, 2])

    cv2.imwrite('{}/sampled_Pano_theta{}_phi{}_fov{}.png'.format(args.output_folder, '%04d'%theta, '%04d'%phi, '%03d'%fov), sampled_Pano)