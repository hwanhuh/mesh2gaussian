from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CXX_ARGS = ['-D_GLIBCXX_USE_CXX11_ABI=0']
NVCC_ARGS = ['--compiler-options', '-D_GLIBCXX_USE_CXX11_ABI=0']

setup(
    name='mesh_to_gs',
    version='0.1.0',
    packages=['mesh_to_gs'],
    ext_modules=[
        CUDAExtension(
            '_C',  
            sources=[
                'csrc/bindings.cpp',
                'csrc/mesh_to_gs_kernel.cu',
            ],
            extra_compile_args={
                'cxx': CXX_ARGS,
                'nvcc': NVCC_ARGS
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)