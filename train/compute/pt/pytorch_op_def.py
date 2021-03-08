import torch
from typing import Dict, Set, Tuple, List, Any, Callable, Iterable, Type

def _NOT_IMPLEMENTED_(*args, **kwargs):
    raise NotImplementedError("Operator not implemented")

pytorch_dtype_map:Dict[str, torch.dtype]  = {
    "float": torch.float,
    "double": torch.double,
    "int": torch.int,
    "long": torch.long
}

class Operator(object):
    def __init__(self, arg_indices:List[int] = [], kwarg_map:Dict[str,int] = {}):
        self.arg_indices = arg_indices
        self.kwarg_map = kwarg_map

    def get_arg_indices(self):
        return self.arg_indices

    def get_kwarg_map(self):
        return self.kwarg_map

# Inplace ops is called in the form of tensor.op(args), we convert it
# to a regular function call with "getattr(tensor, op)(args)"
class InPlaceOpByName(Operator):
    def __init__(self, name:str, arg_indices:List[int] = [], kwarg_map:Dict[str,int] = {}):
        super(InPlaceOpByName, self).__init__(arg_indices, kwarg_map)
        self.name:str = name

    def __call__(self, *args, **kwargs):
        # The first arg is assume to be the inplace value, pass on the rest of
        # the args to the callable.
        getattr(args[0], self.name)(*args[1:], **kwargs)

# Callable ops are ops can be called in the form of op(*args, **kwargs)
class CallableOp(Operator):
    def __init__(self, func:Callable, arg_indices:List[int] = [], kwarg_map:Dict[str,int] = {}):
        super(CallableOp, self).__init__(arg_indices, kwarg_map)
        self.func:Callable = func

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


def get_pytorch_ops() -> Dict[str, Operator]:
    ops_map:Dict[str, Operator] = {
        "Optimizer.step#FusedLAMB.step": None,
        # _embedding_bag_backward(const Tensor &grad, const Tensor &indices,
        #                       const Tensor &offsets,
        #                       const Tensor &offset2bag,
        #                       const Tensor &bag_size_,
        #                       const Tensor &max_indices_,
        #                       int64_t num_weights,
        #                       bool scale_grad_by_freq, int64_t mode,
        #                       bool sparse,
        #                       const Tensor& per_sample_weights)
        "aten::_embedding_bag_backward":  None,
        "aten::add": CallableOp(torch.add, [0, 1]),
        "aten::add_": InPlaceOpByName("add_", [0, 1]),
        "aten::baddbmm": CallableOp(torch.baddbmm, [0,1,2], {"beta":3, "alpha":4}),
        "aten::binary_cross_entropy_with_logits": CallableOp(torch.nn.functional.binary_cross_entropy_with_logits),
        "aten::binary_cross_entropy_with_logits_backward": None,
        "aten::bmm": CallableOp(torch.bmm),
        "aten::cat": CallableOp(torch.cat),
        "aten::clamp_": InPlaceOpByName("clamp_"),
        "aten::conj": None,
        "aten::contiguous": None,
        "aten::copy_": None,
        "aten::cumsum": None,
        "aten::detach": None,
        "aten::detach_": None,
        "aten::div": None,
        "aten::embedding_bag": None, # embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
        "aten::empty": None,
        "aten::empty_like": None,
        "aten::expand": None,
        "aten::flatten": None,
        "aten::gather": None,
        "aten::gather_backward": None,
        "aten::layer_norm": None,
        "aten::le": None,
        "aten::log": None,
        "aten::matmul": CallableOp(torch.matmul),
        "aten::mean": CallableOp(torch.mean, [0]),
        "aten::mm": CallableOp(torch.mm),
        "aten::mul": CallableOp(torch.mul),
        "aten::narrow": None,
        "aten::native_layer_norm_backward": None,
        "aten::neg": None,
        "aten::new_empty_strided": None,
        "aten::ones_like": None,
        "aten::permute": None,
        "aten::pin_memory": None,
        "aten::record_stream": None,
        "aten::relu": CallableOp(torch.nn.functional.relu),
        "aten::reshape": CallableOp(torch.reshape),
        "aten::rsub": None,
        "aten::select": None,
        "aten::select_backward": None,
        "aten::set_": None,
        "aten::sigmoid": None,
        "aten::sigmoid_backward": None,
        "aten::slice": None,
        "aten::split_with_sizes": None,
        "aten::squeeze": None,
        "aten::sub": None,
        "aten::sum": None,
        "aten::t": None,
        "aten::tanh": None,
        "aten::tanh_backward": None,
        "aten::threshold_backward": None,
        "aten::to": None,
        "aten::transpose": None,
        "aten::unsqueeze": None,
        "aten::view": None,
        "aten::where": None,
        "aten::zero_": None,
        "aten::zeros": None,
        "aten::zeros_like": None,
        "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__": None,
        "fb::asynchronous_complete_cumsum": None,
        "fb::asynchronous_exclusive_cumsum": None,
        "fb::offsets_range": None,
        "fb::permute_pooled_embs_auto_grad": None,
        "fb::permute_sparse_features": None,
        "fb::split_embedding_codegen_lookup_rowwise_adagrad_function": None,
    }
    return ops_map
