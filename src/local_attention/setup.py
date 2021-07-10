from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# requires: python setup.py install

# CppExtension是setuptools的便利包装.Extension传递正确的include路径并将扩展语言设置为c++
# BuildExtension执行许多必需的配置步骤并检查，并且在混合c++ / CUDA扩展的情况下混合编译
setup(
    name='local_att', 
    ext_modules=[
        CUDAExtension('local_att', [
            'local_attention.cpp',
            'local_attention_cuda.cu', 
        ],
        extra_compile_args={'cxx': [],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)