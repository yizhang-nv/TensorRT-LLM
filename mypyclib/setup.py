from setuptools import setup
from mypyc.build import mypycify
import os

os.environ["TRTLLM_BUILD_MYPYCLIB"] = "1"

setup(
    name="mypyclib",
    version="0.1.0",
    packages=["mypyclib"],
    ext_modules=mypycify(["mypyclib/__init__.py", "mypyclib/sampler_utils.py", "mypyclib/copy_indices.py"]),
)
