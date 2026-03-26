
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

##############################
# basic operations
##############################

def abs_to_rel(traj_abs: torch.Tensor) -> torch.Tensor:
    """
    args: traj_abs (B, N, 3) - (x, y, theta)
    return: traj_rel (B, N-1, 3) - (dx, dy, dtheta)
    """
    single_sample = False
    if len(traj_abs.shape) == 2:
        single_sample = True
        traj_abs = traj_abs.unsqueeze(0)  # (1, N, 3)
    
    prev = traj_abs[:, :-1, :]  # (B, N-1, 3)
    next = traj_abs[:, 1:, :]    # (B, N-1, 3)
    
    dx = next[:, :, 0] - prev[:, :, 0]  # (B, N-1)
    dy = next[:, :, 1] - prev[:, :, 1]  # (B, N-1)
    theta_rel = next[:, :, 2] - prev[:, :, 2]
    
    # rotate
    theta_prev = prev[:, :, 2]
    cos_theta = torch.cos(-theta_prev)
    sin_theta = torch.sin(-theta_prev)
    x_rel = dx * cos_theta - dy * sin_theta
    y_rel = dx * sin_theta + dy * cos_theta

    # normalize to [-pi, pi]
    # theta_rel = torch.atan2(torch.sin(theta_rel), torch.cos(theta_rel))  
    theta_rel = (theta_rel + torch.pi) % (2 * torch.pi) - torch.pi
    
    traj_rel = torch.stack([x_rel, y_rel, theta_rel], dim=-1)

    if single_sample:
        traj_rel = traj_rel.squeeze(0)
    return traj_rel

def rel_to_abs(traj_rel: torch.Tensor) -> torch.Tensor:
    """
    reconstruct absolute trajectory
    args: traj_rel (B, N-1, 3)
    return: traj_abs (B, N, 3)
    """
    single_sample = False
    if len(traj_rel.shape) == 2:
        single_sample = True
        traj_rel = traj_rel.unsqueeze(0)  # (1, N, 3)
    B, N_minus_1, _ = traj_rel.shape
    N = N_minus_1 + 1
    
    # add (0, 0, 0) at the start
    traj_abs = torch.zeros(B, N, 3, device=traj_rel.device)
    
    for i in range(1, N):
        traj_abs[:, i, 2] = traj_abs[:, i-1, 2] + traj_rel[:, i-1, 2]
        
        theta_i = traj_abs[:, i-1, 2]
        cos_theta = torch.cos(theta_i)
        sin_theta = torch.sin(theta_i)
        x_rel = traj_rel[:, i-1, 0]
        y_rel = traj_rel[:, i-1, 1]
        
        dx = x_rel * cos_theta - y_rel * sin_theta
        dy = x_rel * sin_theta + y_rel * cos_theta
        
        traj_abs[:, i, 0] = traj_abs[:, i-1, 0] + dx
        traj_abs[:, i, 1] = traj_abs[:, i-1, 1] + dy
    
    if single_sample:
        traj_abs = traj_abs.squeeze(0)
    return traj_abs

def abs_to_point_frame(traj_abs: torch.Tensor, point_in_global: torch.Tensor) -> torch.Tensor:
    """
    args:
        traj_abs (B, N, 3) - absolute trajectory (x, y, theta)
        point_in_global (B, 3) - reference point in global coordinates (x, y, theta)
    return: traj_new (B, N, 3) - trajectory under reference coordinate system
    """
    single_sample = False
    if len(traj_abs.shape) == 2:
        single_sample = True
        traj_abs = traj_abs.unsqueeze(0)  # (1, N, 3)
        point_in_global = point_in_global.unsqueeze(0)

    x0 = point_in_global[:, 0].unsqueeze(1)  # (B, 1)
    y0 = point_in_global[:, 1].unsqueeze(1)
    theta0 = point_in_global[:, 2].unsqueeze(1)
    
    dx = traj_abs[:, :, 0] - x0  # (B, N)
    dy = traj_abs[:, :, 1] - y0  # (B, N)
    
    cos_theta0 = torch.cos(theta0)  # (B, 1)
    sin_theta0 = torch.sin(theta0)  # (B, 1)
    
    x_rel = dx * cos_theta0 + dy * sin_theta0
    y_rel = -dx * sin_theta0 + dy * cos_theta0
    
    theta_rel = traj_abs[:, :, 2] - theta0
    # theta_rel = torch.atan2(torch.sin(theta_rel), torch.cos(theta_rel))
    theta_rel = (theta_rel + torch.pi) % (2 * torch.pi) - torch.pi
    
    traj_new = torch.stack([x_rel, y_rel, theta_rel], dim=-1)
    
    if single_sample:
        traj_new = traj_new.squeeze(0)
    return traj_new

##############################
# functions
##############################

def construct_relative_trajectory(traj_abs: torch.Tensor):
    """
    past: trajectory[:NUM_HISTORY_FRAMES]
    future: trajectory[NUM_HISTORY_FRAMES-1:NUM_HISTORY_FRAMES+n]
    """
    return abs_to_rel(traj_abs)

def reconstruct_past_trajectory(traj_rel: torch.Tensor):
    traj_abs = rel_to_abs(traj_rel)
    traj_abs = abs_to_point_frame(traj_abs, traj_abs[..., -1, :])
    return traj_abs[..., :-1, :]

def reconstruct_future_trajectory(traj_rel: torch.Tensor):
    return rel_to_abs(traj_rel)[..., 1:, :]
    
##############################
# processor
##############################

class ActionProcessor:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def normalize(self, x: torch.Tensor):
        x = construct_relative_trajectory(x)
        return self.norm(x)

    def denormalize(self, x: torch.Tensor, future=True):
        x = self.denorm(x)
        if future:
            return reconstruct_future_trajectory(x)
        else:
            return reconstruct_past_trajectory(x)
