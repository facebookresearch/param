import logging


def standardize_comm_name(name: str) -> str:
    """
    Converts informal collective communication names to their standard internal representations,
    logging a warning if the name is not recognized.

    Args:
        name (str): The informal name of the collective communication.

    Returns:
        str: The standardized communication name or the original name if no standard form is found.
    """
    name_aliases = {
        "alltoall": "all_to_all",
        "alltoallv": "all_to_allv",
        "alltoallbase": "all_to_allv",
        "alltoallsingle": "all_to_all_single",
        "allreduce": "all_reduce",
        "allgather": "all_gather",
        "allgatherbase": "all_gather_base",
        "reducescatter": "reduce_scatter",
        "reducescatterbase": "reduce_scatter_base",
        "recvanysource": "recv",
    }

    logger = logging.getLogger(__name__)
    normalized_name = "".join(char for char in name.lower() if char.isalpha())

    if normalized_name in name_aliases:
        return name_aliases[normalized_name]
    else:
        logger.warning(
            f"Unrecognized collective communication name '{name}'. Returning original name."
        )
        return name
