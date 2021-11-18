from setuptools import setup

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
    version_config={
        "template": "{full_sha}",
        "dev_template": "{full_sha}.post{ccount}+git.{sha}",
        "dirty_template": "{tag}.post{ccount}+git.{sha}.dirty",
        "starting_version": "{full_sha}",
        "version_callback": None,
        "version_file": None,
        "count_commits_from_version_file": False,
        "branch_formatter": None,
        "sort_by": None,
    },
    setup_requires=["setuptools-git-versioning"],
    author="Louis Feng",
    author_email="lofe@fb.com",
    packages=packages,
    package_dir=package_dir_map,
)
