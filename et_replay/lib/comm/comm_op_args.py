from typing import Dict


class CommOpArgs:
    """
    This class contains all of the args that we can use to perform a single
    collective.

    Attributes:
        comms (str): Name of collective.
        id (int): Current trace object ID.
        req (int): Request ID of collective to map to wait operation.
        in_msg_size (int): Size of input tensor.
        out_msg_size (int): Size of output tensor.
        dtype (str): Data type of tensor values.
        in_split (list): List of input split sizes for rank across current
                        process group.
        out_split (list): List of output split sizes for ranks across current
                         process group.
        startTimeNs (int): Start time of current collective.
        pg_id (int): Unique identifier for the process group this collective
                    will use.
        group_ranks (list): Global ranks of the process group, this is used with
                           PG init.
        world_size (int): World size of current process group.
        marker_stack (list): Current markers that this collective is a part of.
        root (int): Used to determine if collective is src or dst.
        src_rank (int): Src rank of a send/recv op.
        dst_rank (int): Dst rank of a send/recv op.
        batch_p2p (bool): Flag for batch point-to-point communication.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize arguments used for comm replay.
        """
        self.comms = kwargs.get("comms")
        self.id = kwargs.get("id")
        self.req = kwargs.get("req")
        self.in_msg_size = kwargs.get("in_msg_size")
        self.out_msg_size = kwargs.get("out_msg_size")
        self.dtype = kwargs.get("dtype")
        self.in_split = kwargs.get("in_split")
        self.out_split = kwargs.get("out_split")
        self.startTimeNs = kwargs.get("startTimeNs")
        self.pg_id = kwargs.get("pg_id")
        self.group_ranks = kwargs.get("group_ranks")
        self.world_size = kwargs.get("world_size")
        self.marker_stack = kwargs.get("marker_stack")
        self.root = kwargs.get("root")
        self.src_rank = kwargs.get("src_rank")
        self.dst_rank = kwargs.get("dst_rank")
        self.batch_p2p = kwargs.get("use_batch")

    def to_dict(self) -> Dict:
        """
        Convert CommArgs to dictionary for storing in json.

        Returns:
            dict: Dictionary containing the comms metadata.
        """
        comm_data = {}
        if self.comms is not None:
            comm_data["comms"] = self.comms
        if self.req is not None:
            comm_data["req"] = self.req
        if self.in_msg_size is not None:
            comm_data["in_msg_size"] = self.in_msg_size
            comm_data["out_msg_size"] = self.out_msg_size
            comm_data["dtype"] = self.dtype
        if self.in_split is not None:
            comm_data["in_split"] = self.in_split
        if self.out_split is not None:
            comm_data["out_split"] = self.out_split
        if self.startTimeNs is not None:
            comm_data["startTime_ns"] = self.startTimeNs
        if self.pg_id is not None:
            comm_data["pg_id"] = self.pg_id
        if self.world_size is not None:
            comm_data["world_size"] = self.world_size
        if self.root is not None:
            comm_data["root"] = self.root

        return comm_data

    def __eq__(self, other: object) -> bool:
        """
        Used for testing. Check if two comms are equal.

        Args:
            other (object): Another CommArgs object to compare.

        Returns:
            bool: True if equal, else False.
        """
        if not isinstance(other, CommOpArgs):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        """
        Return the string representation of the CommArgs object.

        Returns:
            str: String representation of the object.
        """
        return f"CommOpArgs({self.__dict__})"
