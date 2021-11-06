from setuptools import setup

package_base = "param_bench.train.compute.python"

# Only list the top level packages and their dir mapping:
# "install_destination_package_path": "source_dir_path"
package_dir_map = {
  f"{package_base}": ".",
  f"{package_base}.lib": "lib",
  f"{package_base}.lib.pytorch": "lib/pytorch",
  f"{package_base}.workloads.pytorch": "workloads/pytorch",
}

packages = list(package_dir_map)

setup(
    name="parambench-train-compute",
    version="0.0.1",
    author="Louis Feng",
    author_email="lofe@fb.com",
    packages=packages,
    package_dir=package_dir_map
)
