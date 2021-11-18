from setuptools import setup

from lib import __version__


def main():
    package_base = "param_bench.train.compute.python"

    # List the packages and their dir mapping:
    # "install_destination_package_path": "source_dir_path"
    package_dir_map = {
        f"{package_base}": ".",
        f"{package_base}.examples": "examples",
        f"{package_base}.examples.pytorch": "examples/pytorch",
        f"{package_base}.lib": "lib",
        f"{package_base}.lib.pytorch": "lib/pytorch",
        f"{package_base}.pytorch": "pytorch",
        f"{package_base}.test": "test",
        f"{package_base}.test.pytorch": "test/pytorch",
        f"{package_base}.tools": "tools",
        f"{package_base}.workloads": "workloads",
        f"{package_base}.workloads.pytorch": "workloads/pytorch",
    }

    packages = list(package_dir_map)

    setup(
        name="parambench-train-compute",
        version=__version__,
        python_requires=">=3.8",
        author="Louis Feng",
        author_email="lofe@fb.com",
        url="https://github.com/facebookresearch/param",
        packages=packages,
        package_dir=package_dir_map,
    )


if __name__ == "__main__":
    main()
