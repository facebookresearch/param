import os

from et_replay.utils import get_first_positive_env_value, read_mpi_env_vars


def test_get_first_positive_env_value(mocker):
    # Mocking os.environ.get
    mocker.patch.dict(
        os.environ, {"TEST_ENV_1": "5", "TEST_ENV_2": "-1", "TEST_ENV_3": "3"}
    )

    assert get_first_positive_env_value(["TEST_ENV_1", "TEST_ENV_2", "TEST_ENV_3"]) == 5
    assert get_first_positive_env_value(["TEST_ENV_2", "TEST_ENV_3"]) == 3
    assert get_first_positive_env_value(["TEST_ENV_2"]) == -1
    assert get_first_positive_env_value(["TEST_ENV_4"], default=10) == 10


def test_read_mpi_env_vars(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "MV2_COMM_WORLD_SIZE": "16",
            "MPI_LOCALNRANKS": "4",
            "PMI_RANK": "2",
            "SLURM_LOCALID": "1",
        },
    )

    world_size, local_size, global_rank, local_rank = read_mpi_env_vars()

    assert world_size == 16
    assert local_size == 4
    assert global_rank == 2
    assert local_rank == 1

    mocker.patch.dict(os.environ, {}, clear=True)

    world_size, local_size, global_rank, local_rank = read_mpi_env_vars()

    assert world_size == -1
    assert local_size == -1
    assert global_rank == -1
    assert local_rank == -1
