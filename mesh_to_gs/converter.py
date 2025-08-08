import torch
import numpy as np
import torch.nn as nn
import os
import math
from math import pi, erf
from plyfile import PlyData, PlyElement

import _C
from .gaussian_model import GaussModel

C0 = 0.28209479177387814

# x 1.17?
def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0

def mesh_to_gs_cuda(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray = None,
    uvs: np.ndarray = None,
    texture: np.ndarray = None,
    R: int = 512,
    k: float = 1.6,
    device: str = 'cuda',
    max_sh_degree: int = 0,
    epsilon: float = 1e-3,
    min_scale: float = 1.0/2048,
    max_scale: float = 1.0/8
):
    """
    Converts a mesh to a Gaussian Splatting model using a high-performance C++/CUDA backend.
    """
    if device != 'cuda':
        raise ValueError("This implementation only supports CUDA devices.")

    # 1. Prepare tensors and move to GPU
    verts_tensor = torch.from_numpy(vertices.astype(np.float32)).to(device)
    faces_tensor = torch.from_numpy(faces.astype(np.int32)).to(device)

    vc_tensor = torch.empty(0, device=device)
    if vertex_colors is not None:
        vc_tensor = torch.from_numpy(vertex_colors.astype(np.float32)).to(device)

    uvs_tensor = torch.empty(0, device=device)
    if uvs is not None:
        uvs_tensor = torch.from_numpy(uvs.astype(np.float32)).to(device)
    
    tex_tensor = torch.empty(0, device=device)
    if texture is not None:
        # Texture data needs to be in a channel-first format for some CUDA operations
        # For tex2D, HWC is fine. Let's assume it's float32 in [0,1]
        tex_data = texture.astype(np.float32)
        if tex_data.max() > 1.1:
            tex_data = tex_data / 255.0
        tex_tensor = torch.from_numpy(tex_data).to(device)

    # 2. Call the C++/CUDA function
    xyz, colors, scales, rots, opacities = _C.convert(
        verts_tensor, faces_tensor, vc_tensor, uvs_tensor, tex_tensor,
        float(R), float(k), epsilon, min_scale, max_scale
    )
    
    if xyz.numel() == 0:
        raise RuntimeError('No Gaussians produced; try lowering R or increasing k/min counts.')

    gm = GaussModel(sh_degree=max_sh_degree)
    
    sh_dim = (max_sh_degree + 1) ** 2
    features = torch.zeros((xyz.shape[0], sh_dim, 3), dtype=torch.float32, device=device)
    features[:, 0, :] = rgb_to_sh(colors)
    
    gm._xyz = torch.nn.Parameter(xyz.requires_grad_(True))
    gm._features_dc = torch.nn.Parameter(features[:, 0:1, :].transpose(1, 2).contiguous().requires_grad_(True))
    gm._features_rest = torch.nn.Parameter(features[:, 1:, :].transpose(1, 2).contiguous().requires_grad_(True))
    gm._scaling = torch.nn.Parameter(torch.log(scales).requires_grad_(True))
    gm._rotation = torch.nn.Parameter(rots.requires_grad_(True))
    gm._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
    gm.max_radii2D = torch.zeros((gm._xyz.shape[0]), device=device)
    
    return gm