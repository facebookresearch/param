import time

__base_version__ = "1.0.0"


def __generate_git_param_train_compute_version():
    # git hash
    commit_version = "+git"
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        commit_version = f"{commit_version}.{repo.head.object.hexsha}"
    except Exception:
        pass

    timestamp = int(time.time())
    commit_version = f"{commit_version}.{timestamp}"
    return f"{__base_version__}{commit_version}"


def __generate_fbcode_param_train_compute_version():
    # Meta build hash
    commit_version = "+fbcode"
    try:
        from __manifest__ import fbmake

        if fbmake["revision"]:
            commit_version = f"{commit_version}.{fbmake['revision']}"
        if fbmake["time"]:
            commit_version = f"{commit_version}.{fbmake['epochtime']}"
        else:
            timestamp = int(time.time())
            commit_version = f"{commit_version}.{timestamp}"
    except Exception:
        commit_version = "+local"

    return f"{__base_version__}{commit_version}"


def __get_version():
    # First try to get the version from setup.py generated _version.py file.
    try:
        from ._version import __param_train_compute_version

        return __param_train_compute_version
    except Exception:
        pass
    # If failed try to get fbcode build version.
    return __generate_fbcode_param_train_compute_version()


__version__ = __get_version()
