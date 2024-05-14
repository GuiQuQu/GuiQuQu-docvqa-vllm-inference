import torch
import os
import subprocess
import re
print("torch version:",torch.__version__)
print("torch cuda version",torch.version.cuda)
print("torch cuda is available:",torch.cuda.is_available())

import torch.utils
import torch.utils.cpp_extension as ex

CUDA_HOME = ex.CUDA_HOME
print("CUDA_HOME:",ex.CUDA_HOME)
nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
SUBPROCESS_DECODE_ARGS = ()
cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)

print("local cuda_version:",cuda_version.group(1))