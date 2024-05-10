from setuptools import setup


def main():
    package_base = "param_bench.train.comms.pt"

    # List the packages and their dir mapping:
    # "install_destination_package_path": "source_dir_path"
    package_dir_map = {
        f"{package_base}": ".",
    }

    packages = list(package_dir_map)

    setup(
        name="parambench-train-comms",
        python_requires=">=3.8",
        author="Louis Feng",
        author_email="lofe@fb.com",
        url="https://github.com/facebookresearch/param",
        packages=packages,
        package_dir=package_dir_map,
    )


if __name__ == "__main__":
    main()
