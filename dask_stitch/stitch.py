import numpy as np
import dask.array as da
import copy
from itertools import product


def weight_block(block, blocksize, overlap, block_info=None):
    """
    """

    # determine which faces need linear weighting
    core, pad_ones, pad_linear = [], [], []
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']
    for i in range(3):

        # get core shape and pad sizes
        o = overlap[i]  # shorthand
        c = blocksize[i] - 2*o + 2
        p_ones, p_linear = [0, 0], [2*o-1, 2*o-1]
        if block_index[i] == 0:
            p_ones[0], p_linear[0] = 2*o-1, 0
        if block_index[i] == block_grid[i] - 1:
            p_ones[1], p_linear[1] = 2*o-1, 0
        core.append(c)
        pad_ones.append(tuple(p_ones))
        pad_linear.append(tuple(p_linear))

    # create weights core
    weights = da.ones(core, dtype=np.float32)

    # extend weights
    weights = da.pad(
        weights, pad_ones, mode='constant', constant_values=1,
    )
    weights = da.pad(
        weights, pad_linear, mode='linear_ramp', end_values=0,
    )

    # block may be a vector field
    # conditional is too general, but works for now
    if weights.ndim != block.ndim:
        weights = weights[..., None]

    # multiply data by weights and return
    return da.multiply(block, weights)


def merge_overlaps(block, overlap):
    """
    """

    o = overlap  # shorthand
    core = [slice(2*x, -2*x) for x in o]
    result = np.require(block[tuple(core)], requirements='W')

    # faces
    for ax in range(3):
        # the left side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(0, o[ax])
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(0, o[ax])
        result[tuple(slc1)] += block[tuple(slc2)]
        # the right side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(-1*o[ax], None)
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(-1*o[ax], None)
        result[tuple(slc1)] += block[tuple(slc2)]

    # edges
    for edge in product([0, 1], repeat=2):
        for ax in range(3):
            oo = np.delete(o, ax)
            left = [slice(None, oe) for oe in oo]
            right = [slice(-1*oe, None) for oe in oo]
            slc1 = [l if e == 0 else r for l, r, e in zip(left, right, edge)]
            slc2 = copy.deepcopy(slc1)
            slc1.insert(ax, slice(None, None))
            slc2.insert(ax, core[ax])
            result[tuple(slc1)] += block[tuple(slc2)]

    # corners
    for corner in product([0, 1], repeat=3):
        left = [slice(None, oe) for oe in o]
        right = [slice(-1*oe, None) for oe in o]
        slc = [l if c == 0 else r for l, r, c in zip(left, right, corner)]
        result[tuple(slc)] += block[tuple(slc)]

    return result


def stitch_blocks(blocks, blocksize, overlap):
    """
    """

    # blocks may be a vector fields
    # conditional is too general, but works for now
    if blocks.ndim != len(blocksize):
        blocksize = list(blocksize) + [3,]
        overlap = tuple(overlap) + (0,)

    # weight block edges
    weighted_blocks = da.map_blocks(
        weight_block, blocks,
        blocksize=blocksize[:3],
        overlap=overlap[:3],
        dtype=np.float32,
    )

    # stitch overlap regions, return
    return da.map_overlap(
        merge_overlaps, weighted_blocks,
        overlap=overlap[:3],
        depth=overlap,
        boundary=0.,
        trim=False,
        dtype=np.float32,
        chunks=blocksize,
    )

