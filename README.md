# Mesh2GaussianSplatting

**Fast Polygonal Mesh to Gaussian Splatting Converter.**
![](./teaser.png)
This project provides a CUDA-accelerated package for converting 3D meshes into Gaussian Splatting representations. 
It also includes a pure Python implementation for reference and comparison.

For ~50K faces mesh: 

| CUDA | Python |
| --- | --- |
| 0.44 sec | 82 sec |

## Features

- Utilizes CUDA for high-performance conversion of meshes to Gaussian splats.
- Includes a pure Python implementation for understanding the conversion logic and for use in environments without a CUDA-compatible GPU.
- Supports meshes with vertex colors or UV-mapped textures.

## Installation

To install the CUDA package, clone this repository and run the following command from the root directory.
Note: I only tested it with torch2.5.1+cu121.

```bash
git clone https://github.com/hwanhuh/mesh2gaussian.git
cd mesh2gaussian
pip install torch plyfile trimesh 
pip install . --no-cache-dir --verbose --no-build-isolation
```

## Usage

### Python (CUDA)

The following example demonstrates how to use the CUDA-accelerated converter.

```bash
python mesh2gs.py 'mesh.glb' 'output.ply'
```

### Pure Python

For comparison or use in non-CUDA environments, you can use the pure Python implementation:

```bash
python mesh2gs_python.py 'mesh.glb' 'output.ply'
```

## How it Works

The conversion process samples points on the surface of the input mesh and represents them as anisotropic 3D Gaussians. 
The density and scale of these Gaussians are determined by the local geometry of the mesh, aiming to create a faithful representation of the original surface.
The CUDA implementation significantly accelerates this process by parallelizing the sampling and Gaussian generation steps on the GPU.

- I use the GaussModel class implemented in [torch-splatting](https://github.com/hbb1/torch-splatting)
