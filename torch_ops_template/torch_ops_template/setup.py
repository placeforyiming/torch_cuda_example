from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="torch_ops_template",
    packages=['torch_ops_template'],
    ext_modules=[
        CUDAExtension(
            name="torch_ops_template._C",
            sources=[
            "src/torch_template_impl.cu",
            #"src/forward.cu",
            #"src/backward.cu",
            "torch_ops_template.cu",
            "ext.cpp"],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-g", "-G", "-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
