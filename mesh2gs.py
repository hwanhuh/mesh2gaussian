import numpy as np
import math
import torch
import torch.nn as nn
import trimesh
import argparse
import os
from math import ceil, pi, erf
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from mesh_to_gs import mesh_to_gs_cuda

def main():
    parser = argparse.ArgumentParser(description="Convert a 3D mesh file (GLB/GLTF) to a Gaussian Splatting .ply file.")
    parser.add_argument("input_path", type=str, help="Path to the input GLB/GLTF file.")
    parser.add_argument("output_path", type=str, help="Path to save the output .ply file.")
    parser.add_argument("--resolution", "-R", type=int, default=512, help="Target resolution constant 'R' for sampling density.")
    parser.add_argument("--density_factor", "-k", type=float, default=2.0, help="Density factor 'k' for overlapping Gaussians.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for processing (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--sh_degree", type=int, default=0, help="Maximum degree of spherical harmonics (0 for simple RGB).")
    parser.add_argument("--preserve_original_coords", action="store_true", help="Preserve original GLB coordinates and scale.")
    args = parser.parse_args()

    print(f"Loading mesh from: {args.input_path}")
    
    try:
        mesh = trimesh.load(args.input_path, force='mesh', process=True)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    original_vertices = mesh.vertices.copy()
    original_center = original_vertices.mean(axis=0)
    original_max_extent = np.max(np.linalg.norm(original_vertices - original_center, axis=1))
    
    if not args.preserve_original_coords:
        print("Normalizing mesh...")
        vertices = original_vertices.copy()
        vertices -= original_center
        scale_factor = original_max_extent
        if scale_factor > 1e-6:
            vertices /= scale_factor
        else:
            scale_factor = 1.0
    else:
        vertices = original_vertices.copy()
        scale_factor = 1.0
    
    faces = mesh.faces
    print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces.")

    uvs = None
    texture = None
    vertex_colors = None

    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
        material = mesh.visual.material
        if hasattr(material.baseColorTexture, 'convert'):
            print("Found texture. Converting to numpy array.")
            texture = np.array(material.baseColorTexture.convert("RGB"))
            uvs = mesh.visual.uv
            print(f"Texture shape: {texture.shape}, UVs shape: {uvs.shape}")
    
    if texture is None and hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0

    try:
        gs_model = mesh_to_gs_cuda(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            uvs=uvs,
            texture=texture,
            R=args.resolution,
            k=args.density_factor,
            device=args.device,
            max_sh_degree=args.sh_degree
        )
    except RuntimeError as e:
        print(f"Error during conversion: {e}")
        return

    if not args.preserve_original_coords:
        with torch.no_grad():
            gs_model._xyz.data = gs_model._xyz.data * scale_factor + torch.tensor(original_center, 
                                                                                 device=gs_model._xyz.device, 
                                                                                 dtype=gs_model._xyz.dtype)
            
    print(f"Saving Gaussian Splatting model to: {args.output_path}")
    gs_model.save_ply(args.output_path)
    
    print("Conversion complete.")
    num_gaussians = gs_model._xyz.shape[0]
    print(f"Generated {num_gaussians} Gaussians.")

if __name__ == "__main__":
    main()