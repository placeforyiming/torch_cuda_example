from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_add",
    include_dirs=["include"],
    packages=["custom_add"],
    ext_modules=[
        CUDAExtension(
            name="custom_add._C",
            sources=["src/custom_add.cpp", "src/cuda/custom_add.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)