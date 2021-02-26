""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
from models import ops


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edge, n_ops) /// torch.Tensor is a multi-dimensional matrix containing of a single data type.
        # torch.topk(input, k): returns the k largest element of the given input tensor along a given dimension.

        edge_max, primitive_indices = torch.topk(edges[:, :], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

def mask_to_str(mask):
    """

    :param mask: Architecture encoded infos
    :return: Genotype string
    """

    mask = mask[0][0][0]
    normal = mask[0:len(mask)//2].reshape((-1, len(PRIMITIVES)))
    reduce = mask[len(mask)//2:].reshape((-1, len(PRIMITIVES)))

    numIntermediateNode = 4
    indexBuffer = 0
    normalList, reduceList = [], []
    for i in range(numIntermediateNode):
        normalList.append(normal[indexBuffer:indexBuffer + i + 2])
        reduceList.append(reduce[indexBuffer:indexBuffer + i + 2])
        indexBuffer += i + 2

    genotype_normal, genotype_reduce = [], []
    for node in normalList:
        bufferList = []
        for j, row in enumerate(node):
            index = np.where(row == 1)[0]
            for k in index:
                bufferList.append((PRIMITIVES[k], j))
        genotype_normal.append(bufferList)

    for node in reduceList:
        bufferList = []
        for j, row in enumerate(node):
            index = np.where(row == 1)[0]
            for k in index:
                bufferList.append((PRIMITIVES[k], j))
        genotype_reduce.append(bufferList)

    concat = range(2, 6)
    genotype = Genotype(genotype_normal, concat, genotype_reduce, concat)

    return genotype