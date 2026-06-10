from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="afno_cuda",
    ext_modules=[
        CUDAExtension(
            name="afno_cuda",
            sources=[
                "afno_cuda.cpp",
                "afno_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
