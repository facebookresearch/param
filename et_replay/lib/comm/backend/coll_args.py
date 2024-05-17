class CollArgs:
    """Class holding object for all the parameters related to a collective operation/experiment."""

    def __init__(self) -> None:
        self.group = None
        self.groups = {}  # {pg_id, pg}
        self.num_pgs = 0
        self.device = {}
        self.world_size = 0
        self.data_type = ""

        self.numIters = 0
        self.numWarmupIters = 0
        self.global_rank = -1
        self.backendFuncs = {}
        self.collective = ""
        self.collectiveId = 0
        self.pt2pt = ""
        self.src_rank = -1
        self.dst_rank = -1
        self.p2pOps = []

        self.reuseTensors = False

        self.batch_size = 0

        self.input_tensor_split = []
        self.output_tensor_split = []

        self.input_tensor = []
        self.output_tensor = []
        self.srcOrDst = -1
        self.asyncOp = -1
        self.dataSize = 0
        self.numElements = 0
        self.waitObj = []
        self.waitObjIds = {}  # mapping of reqID to future of async collectives

        self.input_tensor_split_pair = []
        self.output_tensor_split_pair = []

        self.input_tensor_pair = None
        self.output_tensor_pair = None
        self.dataSize_pair = 0
        self.numElements_pair = 0

        self.all2all_qcomm = None
        self.reducescatter_allgather_qcomm = None
        self.allreduce_qcomm = 32  # TODO: set it as the bitwidth for now until the quantization kernels be supported
        self.reduce_qcomm = 32
        self.quant_threshold = 0
        self.enable_profiler = False

        self.use_ext_dist = False
