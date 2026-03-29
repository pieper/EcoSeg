"""Find out why GrowCut stops expanding — using the actual implementation."""

import torch
import numpy as np
from ecoseg.models.growcut_embedding import growcut_intensity, GrowCutConfig


# Test 1: uniform volume, should label everything
vol = torch.ones(16, 16, 16) * 0.5
seeds = torch.zeros(16, 16, 16, dtype=torch.int32)
seeds[8, 8, 4] = 1
seeds[8, 8, 12] = 2

config = GrowCutConfig(max_iterations=500, convergence_threshold=0.0)
labels, strength = growcut_intensity(vol, seeds, config)

n_unlabeled = (labels == 0).sum().item()
n_fg = (labels == 1).sum().item()
n_bg = (labels == 2).sum().item()
print(f"Uniform volume: unlabeled={n_unlabeled}, fg={n_fg}, bg={n_bg}")

# Check the stuck voxels
unlabeled = labels == 0
labeled = labels > 0

offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
has_labeled_neighbor = torch.zeros_like(unlabeled)
for dz, dy, dx in offsets:
    nz = torch.roll(labeled, shifts=(-dz,-dy,-dx), dims=(0,1,2))
    # Fix boundary
    boundary_ok = torch.ones_like(labeled)
    if dz == -1: boundary_ok[0,:,:] = False
    elif dz == 1: boundary_ok[-1,:,:] = False
    if dy == -1: boundary_ok[:,0,:] = False
    elif dy == 1: boundary_ok[:,-1,:] = False
    if dx == -1: boundary_ok[:,:,0] = False
    elif dx == 1: boundary_ok[:,:,-1] = False
    has_labeled_neighbor |= (nz & boundary_ok)

stuck = unlabeled & has_labeled_neighbor
print(f"Stuck (unlabeled with labeled neighbor): {stuck.sum()}")

# Manual check: can a stuck voxel be attacked?
if stuck.any():
    coords = torch.argwhere(stuck)
    z, y, x = coords[0].tolist()
    print(f"\nChecking stuck voxel ({z},{y},{x}):")
    print(f"  label={labels[z,y,x].item()}, strength={strength[z,y,x].item()}")
    for dz, dy, dx in offsets:
        nz, ny, nx = z+dz, y+dy, x+dx
        if 0 <= nz < 16 and 0 <= ny < 16 and 0 <= nx < 16:
            nl = labels[nz, ny, nx].item()
            ns = strength[nz, ny, nx].item()
            nv = vol[nz, ny, nx].item()
            cv = vol[z, y, x].item()
            fitness = 1.0 - abs(cv - nv)
            attack = fitness * ns
            print(f"  neighbor ({nz},{ny},{nx}): label={nl}, str={ns:.4f}, "
                  f"fitness={fitness:.4f}, attack={attack:.4f}, "
                  f"would_win={attack > strength[z,y,x].item()}")
