# mesh_to_gs.py - generated
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

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def inverse_sigmoid(x, eps=1e-6):
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))

def strip_symmetric(mat):
    return 0.5 * (mat + mat.transpose(-1, -2))

def build_scaling_rotation(scaling, rotation):
    if isinstance(scaling, torch.Tensor):
        N = scaling.shape[0]
        Ls = torch.zeros((N,3,3), device=scaling.device, dtype=scaling.dtype)
        for i in range(N):
            s = scaling[i].cpu().numpy()
            Ls[i] = torch.tensor(np.diag(s), dtype=scaling.dtype)
        return Ls
    else:
        return np.diag(scaling)

class GaussModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int=3, debug=False):
        super(GaussModel, self).__init__()
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.debug = debug

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

def mesh_to_gaussmodel_improved(vertices, faces, vertex_colors=None, uvs=None, texture=None,
                                R=512, k=1.6, color_mode='global', device='cpu',
                                max_sh_degree=3, epsilon=1e-3,
                                min_scale=1.0/2048, max_scale=1.0/8):
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

    A_target = (1.0 / R) ** 2
    p = 1.0 / (2.0 * R)

    Nf = np.ceil(k * (areas / A_target)).astype(int)
    small_mask = areas < 1e-12
    Nf[small_mask] = 0

    xyz_list = []
    color_list = []
    scaling_list = []
    rot_list = []
    opacity_list = []

    def sample_barycentric(n):
        if n == 1:
            return np.array([[1/3,1/3,1/3]], dtype=np.float64)
        r1 = np.random.rand(n)
        r2 = np.random.rand(n)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v
        return np.stack([u, v, w], axis=1)

    for fi in tqdm(range(len(faces)), desc="Converting Mesh to Gaussian", unit=" face"):
        n = Nf[fi]
        if n <= 0:
            continue
        a = v0[fi]; b = v1[fi]; c = v2[fi]
        face_normal = normals[fi]

        bary = sample_barycentric(n)
        positions = bary[:,0:1]*a + bary[:,1:2]*b + bary[:,2:3]*c

        edges = [b - a, c - b, a - c]
        best = None; best_len = -1.0
        for e in edges:
            proj = e - np.dot(e, face_normal) * face_normal
            l = np.linalg.norm(proj)
            if l > best_len:
                best_len = l; best = proj / (l + 1e-12)
        if best is None:
            t = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(t, face_normal)) > 0.9:
                t = np.array([0.0, 1.0, 0.0])
            best = t - np.dot(t, face_normal) * face_normal
            best = best / (np.linalg.norm(best) + 1e-12)
        bitangent = np.cross(face_normal, best); bitangent = bitangent / (np.linalg.norm(bitangent)+1e-12)
        Rmat = np.stack([best, bitangent, face_normal], axis=1)
        uvec = Rmat[:,0]; uvec = uvec / (np.linalg.norm(uvec)+1e-12)
        vtmp = Rmat[:,1] - np.dot(Rmat[:,1], uvec)*uvec; vtmp = vtmp / (np.linalg.norm(vtmp)+1e-12)
        wtmp = np.cross(uvec, vtmp)
        Rmat = np.stack([uvec, vtmp, wtmp], axis=1)
        trace = Rmat[0,0] + Rmat[1,1] + Rmat[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (Rmat[2,1] - Rmat[1,2]) * s
            qy = (Rmat[0,2] - Rmat[2,0]) * s
            qz = (Rmat[1,0] - Rmat[0,1]) * s
        else:
            if Rmat[0,0] > Rmat[1,1] and Rmat[0,0] > Rmat[2,2]:
                s = 2.0 * math.sqrt(1.0 + Rmat[0,0] - Rmat[1,1] - Rmat[2,2])
                qw = (Rmat[2,1] - Rmat[1,2]) / s
                qx = 0.25 * s
                qy = (Rmat[0,1] + Rmat[1,0]) / s
                qz = (Rmat[0,2] + Rmat[2,0]) / s
            elif Rmat[1,1] > Rmat[2,2]:
                s = 2.0 * math.sqrt(1.0 + Rmat[1,1] - Rmat[0,0] - Rmat[2,2])
                qw = (Rmat[0,2] - Rmat[2,0]) / s
                qx = (Rmat[0,1] + Rmat[1,0]) / s
                qy = 0.25 * s
                qz = (Rmat[1,2] + Rmat[2,1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + Rmat[2,2] - Rmat[0,0] - Rmat[1,1])
                qw = (Rmat[1,0] - Rmat[0,1]) / s
                qx = (Rmat[0,2] + Rmat[2,0]) / s
                qy = (Rmat[1,2] + Rmat[2,1]) / s
                qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz], dtype=np.float32)

        edge_lengths = [np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)]
        L = max(edge_lengths) + 1e-12
        Af = max(areas[fi], 1e-12)
        rf = (L**2) / (4.0 * Af + 1e-12)
        rf = max(rf, 1e-6)

        s1 = math.sqrt(rf / (pi * k)) * (1.0 / R)
        s2 = 1.0 / (math.sqrt(pi * k * rf) * R)
        s1 = float(np.clip(s1, min_scale, max_scale))
        s2 = float(np.clip(s2, min_scale, max_scale))

        if vertex_colors is not None:
            c0 = vertex_colors[faces[fi,0]]
            c1 = vertex_colors[faces[fi,1]]
            c2 = vertex_colors[faces[fi,2]]
            cols = bary[:,0:1]*c0 + bary[:,1:2]*c1 + bary[:,2:3]*c2
        elif (uvs is not None) and (texture is not None):
            H, W = texture.shape[0], texture.shape[1]
            uv0 = uvs[faces[fi,0]]; uv1 = uvs[faces[fi,1]]; uv2 = uvs[faces[fi,2]]
            uvs_sample = bary[:,0:1]*uv0 + bary[:,1:2]*uv1 + bary[:,2:3]*uv2
            iu = np.clip((uvs_sample[:,0]*(W-1)).astype(int), 0, W-1)
            iv = np.clip((uvs_sample[:,1]*(H-1)).astype(int), 0, H-1)
            cols = texture[iv, iu].astype(np.float32)
            if cols.max() > 1.1: cols = cols / 255.0
        else:
            cols = 0.5 * np.ones((n,3), dtype=np.float32)

        if color_mode == 'global':
            comp = 1.0 / (pi * (erf(0.5)**2))
            comp_arr = np.full((n,1), comp, dtype=np.float32)
        elif color_mode == 'per_gauss':
            comp_arr = np.zeros((n,1), dtype=np.float32)
            for idx in range(n):
                a1 = p / s1
                a2 = p / s2
                e1 = erf(a1) if a1>0 else 1.0
                e2 = erf(a2) if a2>0 else 1.0
                denom = pi * s1 * s2 * e1 * e2
                if denom <= 0: fact = 1.0
                else: fact = (1.0 / (R*R)) / denom
                comp_arr[idx,0] = float(fact)
        else:
            comp_arr = np.ones((n,1), dtype=np.float32)

        comp_arr = np.clip(comp_arr, 0.5, 1.5)

        for i_sample in range(n):
            xyz_list.append(positions[i_sample].astype(np.float32))
            color_list.append((cols[i_sample] * comp_arr[i_sample]).astype(np.float32))
            scaling_list.append(np.array([s1, s2, epsilon], dtype=np.float32))
            rot_list.append(quat.astype(np.float32))
            opacity_list.append(np.array([inverse_sigmoid(torch.tensor(0.99)).item()], dtype=np.float32))

    if len(xyz_list) == 0:
        raise RuntimeError('No Gaussians produced; try lowering R or increasing k/min counts.')

    xyz = torch.tensor(np.vstack(xyz_list)).float().to(device)
    colors = torch.tensor(np.vstack(color_list)).float().to(device)
    scales = torch.tensor(np.vstack(scaling_list)).float().to(device)
    rots = torch.tensor(np.vstack(rot_list)).float().to(device)
    opac = torch.tensor(np.vstack(opacity_list)).float().to(device)

    sh_dim = (max_sh_degree + 1) ** 2
    features = torch.zeros((xyz.shape[0], 3, sh_dim), dtype=torch.float32, device=device)
    features[:, :3, 0] = RGB2SH(colors)

    gm = GaussModel(sh_degree=max_sh_degree)
    gm._xyz = nn.Parameter(xyz.requires_grad_(True))
    gm._features_dc = nn.Parameter(features[:,:,0:1].transpose(1,2).contiguous().requires_grad_(True))
    gm._features_rest = nn.Parameter(features[:,:,1:].transpose(1,2).contiguous().requires_grad_(True))
    gm._scaling = nn.Parameter(torch.log(scales).requires_grad_(True))
    gm._rotation = nn.Parameter(rots.requires_grad_(True))
    gm._opacity = nn.Parameter(torch.tensor(opac, dtype=torch.float32, device=device).requires_grad_(True))
    gm.max_radii2D = torch.zeros((gm._xyz.shape[0]), device=device)
    return gm

def main():
    parser = argparse.ArgumentParser(description="Convert a 3D mesh file (GLB/GLTF) to a Gaussian Splatting .ply file.")
    parser.add_argument("input_path", type=str, help="Path to the input GLB/GLTF file.")
    parser.add_argument("output_path", type=str, help="Path to save the output .ply file.")
    parser.add_argument("--resolution", "-R", type=int, default=512, help="Target resolution constant 'R' for sampling density.")
    parser.add_argument("--density_factor", "-k", type=float, default=2.0, help="Density factor 'k' for overlapping Gaussians.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for processing (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--sh_degree", type=int, default=0, help="Maximum degree of spherical harmonics (0 for simple RGB).")
    parser.add_argument("--color_mode", type=str, default="per_gauss", choices=['global', 'per_gauss', 'none'],
                        help="Color compensation mode. 'per_gauss' is more accurate but slightly slower.")
    
    args = parser.parse_args()

    print(f"Loading mesh from: {args.input_path}")
    
    try:
        mesh = trimesh.load(args.input_path, force='mesh', process=True)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    if isinstance(mesh, trimesh.Scene):
        print("Scene object loaded, concatenating geometries.")
        mesh = mesh.dump(concatenate=True)

    vertices = mesh.vertices
    faces = mesh.faces
    
    print("Normalizing mesh...")
    center = vertices.mean(axis=0)
    vertices -= center
    max_extent = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= max_extent 
    
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
            uvs[:, 1] = 1.0 - uvs[:, 1]
            print(f"Texture shape: {texture.shape}, UVs shape: {uvs.shape}")
        else:
             print("Texture found but could not be converted.")
    
    if texture is None and hasattr(mesh.visual, 'vertex_colors'):
        print("Using vertex colors.")
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        print(f"Vertex colors shape: {vertex_colors.shape}")


    try:
        gs_model = mesh_to_gaussmodel_improved(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            uvs=uvs,
            texture=texture,
            R=args.resolution,
            k=args.density_factor,
            color_mode=args.color_mode,
            device=args.device,
            max_sh_degree=args.sh_degree
        )
    except RuntimeError as e:
        print(f"Error during conversion: {e}")
        return

    print(f"Saving Gaussian Splatting model to: {args.output_path}")
    gs_model.save_ply(args.output_path)
    
    print("Conversion complete.")
    num_gaussians = gs_model._xyz.shape[0]
    print(f"Generated {num_gaussians} Gaussians.")


if __name__ == "__main__":
    main()