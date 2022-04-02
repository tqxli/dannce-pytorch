"""
Adapted from 
https://github.com/facebookresearch/VideoPose3D/blob/main/common/loss.py
"""
import numpy as np
import torch

def nanmean_infmean(loss):
    valid = (~np.isnan(loss)) & (~np.isposinf(loss)) & (~np.isneginf(loss))
    num_valid = valid.sum()
    if num_valid == 0:
        return 0
    return loss[valid].sum() / num_valid

def euclidean_distance_3D(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return nanmean_infmean(np.sqrt(((target - predicted) ** 2).sum(1)).flatten())

def pa_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    # return np.nanmean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=-1)
    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1)
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = np.mean(np.sum(predicted**2, axis=2, keepdims=True), axis=1, keepdims=True)
    norm_target = np.mean(np.sum(target*predicted, axis=2, keepdims=True), axis=1, keepdims=True)
    scale = norm_target / norm_predicted
    return euclidean_distance_3D(scale * predicted, target)


