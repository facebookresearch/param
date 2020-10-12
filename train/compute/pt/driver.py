# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataset
import pytorch_gemm as kgemm
import pytorch_emb as kemb
import pytorch_linear as klinear

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring the Compute Kernel Performance Using PyTorch"
    )
    parser.add_argument('--warmups', type=int, default=10, help="warmup times")
    parser.add_argument('--steps', type=int, default=100, help="repeat times")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'tpu'], required=True, help='valid devices')

    subparsers = parser.add_subparsers(title='kernels', dest='kernel')
    subparsers.required = True

    parser_gemm = subparsers.add_parser('gemm', help='measure mm performance (m,k)*(k,n)=(m,n)')
    parser_gemm.add_argument('-t', '--dtype', type=str, default="float32")
    parser_gemm.add_argument('-d', '--dataset', choices=['A', 'B', 'C'], default='A')

    parser_emb = subparsers.add_parser('emb', help='measure EmbeddingBag performance')
    parser_emb.add_argument('-d', '--dataset', choices=['A', 'B'], default='A')
    parser_emb.add_argument("--randomseed", type=int, default=0)
    parser_emb.add_argument("--usexlabag", action='store_true', help='use xlabad instead of embeddingbag')

    parser_linear = subparsers.add_parser('linear', help='measure mlp performance')
    parser_linear.add_argument('--optimizer-type', default='sgd', help='Optimizer: SGD', choices=['sgd'])
    parser_linear.add_argument('-t', '--dtype', default='float', help="data type", choices=["float", "float16", "bfloat16"])
    parser_linear.add_argument('-d', '--dataset', choices=['A'], default='A')

    args=parser.parse_args()

    print("Measuring the performance of ", args.kernel, " on device = ", args.device)
    print("Steps = ", args.steps, " warmups = ", args.warmups)
    if args.kernel == 'gemm':
        print("with matrix dataset ", args.dataset, ", Data type: ", args.dtype)
        print(" ")
        if args.dataset == 'A':
            kgemm.run(args, dataset.gemm_A)
        elif args.dataset == 'B':
            kgemm.run(args, dataset.gemm_B)
        else:
            kgemm.run(args, dataset.gemm_C)

    elif args.kernel == 'emb':
        print("with emb dataset ", args.dataset)
        if args.dataset == 'A':
            kemb.run(args, dataset.emb_A)
        elif args.dataset == 'B':
            kemb.run(args, dataset.emb_B)

    else:
        print("with linear dataset ", args.dataset, ", Data type: ", args.dtype)
        if args.dataset == 'A':
            klinear.run(args, dataset.mlp_A)

