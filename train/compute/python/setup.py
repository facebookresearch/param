from setuptools import setup

setup(
    name='param_compute_lib',
    version='0.0.1',
    author='Louis Feng',
    author_email='lofe@fb.com',
    packages=['param_bench.train.compute.lib', "param_bench.train.compute.lib.pytorch",
              "param_bench.train.compute.workloads", "param_bench.train.compute.workloads.pytorch"],
    package_dir={"param_bench.train.compute.lib": "lib", "param_bench.train.compute.workloads": "workloads"}
)