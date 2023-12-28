import cv2
import numpy as np
from typing import Tuple
import os
import time
import plotly.graph_objs as go
'''import torch
from torch_points3d.applications.matching import SuperGlue as SuperGlueApp
from torch_points3d.applications.conf.matching.SuperGlue import SuperGlueConfig'''
from compute_utils import *

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

def rgb_loader(path):
    return np.asarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).astype(np.float64)

def feature_matching_sift(patch1: np.ndarray, patch2: np.ndarray, type_of_matching: str) -> Tuple[np.ndarray, np.ndarray]:

    # Initialize SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()

    # Extract keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(patch1, None)
    kp2, des2 = sift.detectAndCompute(patch2, None)

    if type_of_matching=='BF':
        # Initialize Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match keypoints
        matches = bf.match(des1, des2)

        # Sort matches by distance (in ascending order)
        matches = sorted(matches, key=lambda x: x.distance)
        avg_distance = sum(m.distance for m in matches) / len(matches)

        good_matches = [m for m in matches if m.distance < 0.7 * avg_distance]
        
    elif type_of_matching=='FLANN': # using FLANN matcher
        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(des1, des2, k=2)

        # Sort matches by distance (in ascending order)
        #matches = sorted(matches, key=lambda x: x.distance)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
    else:
        return NotImplementedError('Matching type {} is not implemented'.format(type_of_matching))

    # Apply ratio test to filter matches
    

    # Extract matched keypoints
    #src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return kp1, kp2, good_matches

def get_K(fov, height, width):
    fov_radiance = np.deg2rad(fov)
    focal_length = (width/2) / np.tan(fov_radiance/2)

    cx, cy = width/2.0, height/2.0

    return np.array([[focal_length, 0, cx],
                    [0, focal_length, cy],
                    [0, 0, 1.0]])

# Geometric
def rotation_matrix_from_thetaphi(theta, phi):
    R_lattitude = np.array([[np.cos(phi), 0, -np.sin(phi)],
               [0, 1, 0],
               [np.sin(phi), 0, np.cos(phi)]])
    R_longitude = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
    
    '''patch_center = np.array([1, 0, 0]).reshape([3])
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    target_dir = np.array([x, y, z]).reshape([3])

    rotated_dir = R_longitude @ R_lattitude @ patch_center'''


    return R_longitude @ R_lattitude

def thetaphi_from_rotation_matrix(R):
    # Apply the rotation matrix R to the unit vector along the x-axis
    rotated_dir = R @ np.array([1, 0, 0])

    # Extract the components of the rotated_dir
    x, y, z = rotated_dir

    # Calculate the theta and phi angles
    theta = np.arctan2(y, x)
    phi = np.arcsin(z)

    return theta, phi

def angle_bet_rots(rot1, rot2):
    """Compute the rotation angle between two 3x3 rotation matrices."""
    rel_rot = np.dot(rot1, rot2.T)
    trace = np.trace(rel_rot)
    angle = np.arccos((trace - 1.0) / 2.0)
    
    return np.degrees(angle)

# GPT
def compute_reprojection_error(points1, points2, E):
    # Compute the epipolar lines in the second image
    lines2 = cv2.computeCorrespondEpilines(points1, 1, E)

    # Compute the distances between the points and their epipolar lines
    errors = []
    for i in range(len(points2)):
        x, y = points2[i, 0]
        a, b, c = lines2[i, 0]
        error = abs(a*x + b*y + c) / np.sqrt(a*a + b*b)
        errors.append(error)

    # Compute the mean and standard deviation of the errors
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    return mean_error, std_error

def estimate_rotation(points1, points2):
    # Compute the centroid of the matched 3D points

    # Compute the centered matched 3D points
    # Compute the covariance matrix of the matched 3D points
    covariance_matrix = np.dot(points1.T, points2)

    # Compute the singular value decomposition of the covariance matrix
    U, s, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix from the left singular vectors
    R = np.dot(U, Vt)

    # Ensure that the determinant of the rotation matrix is positive
    if np.linalg.det(R) < 0:
        R *= -1

    # Convert the rotation matrix to a rotation object
    return R

def pix2cam_ball(pts, K):
    # Invert the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Calculate the radius of the sphere
    corner_3d = K_inv @ np.array([0,0,1])
    radius = np.linalg.norm(corner_3d)

    # Add a third coordinate to the pixel points (u, v) to make them homogeneous (u, v, 1)
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))

    # Normalize the pixel coordinates by applying the inverse of the intrinsic matrix K
    pts_normalized = (K_inv @ pts_homogeneous.T).T

    # Calculate the intersection point t
    A = np.sum(pts_normalized[:, :2] ** 2, axis=1) + 1
    t = np.sqrt(radius ** 2 / A)

    # Calculate the 3D coordinates (X, Y, Z) of the pixel in the camera frame
    pts_cam = np.zeros((pts.shape[0], 3))
    pts_cam[:, :2] = pts_normalized[:, :2] * t[:, np.newaxis]
    pts_cam[:, 2] = t

    return pts_cam

if __name__ == "__main__":
    # Data loading
    Pano_root = 'datasets/small_overlap/raw_crops/undist'
    # find pairs
    meta_npy = os.path.join('datasets/small_overlap/metadata/my_sun360/undist/sun_360.npy')
    pairs = np.load(meta_npy, allow_pickle=True).item()

    # read patches and pano
    ld_idx = 300
    img1 = pairs[ld_idx]['img1']
    img2 = pairs[ld_idx]['img2']
    pano = pairs[ld_idx]['pano']

    path1 = os.path.join(Pano_root, img1['path'])
    rotation_x1, rotation_y1 = img1['x'], img1['y']
    image1 = cv2.imread(path1)
    
    path2 = os.path.join(Pano_root, img2['path'])
    rotation_x2, rotation_y2 = img2['x'], img2['y']
    image2 = cv2.imread(path2)

    path_pano = os.path.join(Pano_root, pano['path'])
    pano = cv2.imread(path_pano)

    patch_height, patch_width = image1.shape[:2]

    # now, do a little matching
    start_time = time.time()
    type_matching = 'BF'
    kpt1, kpt2, good_matches = feature_matching_sift(image1, image2, type_matching)
    
    # print duration
    end_time = time.time()
    # print('The SIFT extraction time is {} s'.format(end_time-start_time))
    
    # little visualization
    match_img = cv2.drawMatches(image1, kpt1, image2, kpt2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('matching_results_{}.jpg'.format(type_matching), match_img)

    theta_gt = rotation_x1 - rotation_x2
    phi_gt = rotation_y1 - rotation_y2
    
    #debug
    gt_mat = compute_gt_rmat(torch.FloatTensor([rotation_x1]), 
                             torch.FloatTensor([rotation_y1]), 
                             torch.FloatTensor([rotation_x2]), 
                             torch.FloatTensor([rotation_y2]), 
                             batch_size=1).detach().cpu().numpy().squeeze()
    
    Signs = np.array([[ 1,  -1,  -1],
                  [ -1, 1,  1],
                  [-1,  1,  1]])
    
    aligned_gt_mat = Signs * np.fliplr(np.flipud(gt_mat.T))

    # Normalize the pixel coordinates of the matched keypoints
    Rotmat = rotation_matrix_from_thetaphi(theta=theta_gt, phi=phi_gt)
    angle = angle_bet_rots(np.eye(3), Rotmat)
    theta_gt_degree = np.degrees(theta_gt)
    phi_gt_degree = np.degrees(phi_gt)
    my_K = get_K(fov=90, height=patch_height, width=patch_width)

    pts1 = np.float32([kpt1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kpt2[m.trainIdx].pt for m in good_matches])
    pts1_homo = np.hstack([pts1, np.ones_like(pts1[:,0:1])])
    pts2_homo = np.hstack([pts2, np.ones_like(pts2[:,0:1])])
    # 
    pts1_cam = pix2cam_ball(pts1, my_K)
    pts2_cam = pix2cam_ball(pts2, my_K)

    R_pure = estimate_rotation(pts1_cam, pts2_cam)
    Axis_T = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0]
    ])

    Rot_estimated = (Axis_T @ R_pure @ Axis_T.T).T
    theta_estimated, phi_estimated = thetaphi_from_rotation_matrix(Rot_estimated)
    print('Overall Time Cost is {} s'.format(time.time()-start_time))

    print('GT: Theta {}, Phi {}'.format(theta_gt/np.pi*180, phi_gt/np.pi*180))
    print('Pred: Theta {}, Phi {}'.format(theta_estimated/np.pi*180, phi_estimated/np.pi*180))
    print('Error: Theta {}, Phi {}'.format(abs(theta_estimated-theta_gt)/np.pi*180, abs(phi_estimated-phi_gt)/np.pi*180))



    
    