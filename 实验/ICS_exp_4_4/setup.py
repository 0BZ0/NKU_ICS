from setuptools import setup
from torch.utils import cpp_extension
import os

setup(
    
    name='hsigmoid_extension.so',
    ext_modules=[
        cpp_extension.CppExtension(
    
            'hsigmoid_extension',
            ['hsigmoid.cpp']
        )
    ],
    
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension
    }
)


directory = './'

current_dir = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(directory):
    if filename.endswith(".so"):
        new_filename = "hsigmoid_extension.so"
        os.rename(os.path.join(directory, filename), os.path.join(os.path.dirname(current_dir), new_filename))


print("generate .so PASS!\n")