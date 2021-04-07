import numpy as np
import dask.array as da
import copy
from itertools import product

def weight_block(block, blocksize, block_info=None):
    """
    """

    # compute essential parameters
    overlaps = (np.array(blocksize) // 2).astype(int)

    # determine which faces need linear weighting
    core_shape, pads_ones, pads_linear = [], [], []
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']
    for i in range(3):

        # get core shape and pad sizes
        p = overlaps[i]
        core = 3 if blocksize[i] % 2 else 2
        pad_ones, pad_linear = [0, 0], [2*p-1, 2*p-1]
        if block_index[i] == 0:
            pad_ones[0], pad_linear[0] = 2*p-1, 0
        if block_index[i] == block_grid[i] - 1:
            pad_ones[1], pad_linear[1] = 2*p-1, 0
        core_shape.append(core)
        pads_ones.append(tuple(pad_ones))
        pads_linear.append(tuple(pad_linear))

    # create weights core
    weights = np.ones(core_shape, dtype=np.float32)

    # extend weights
    weights = da.pad(
        weights, pads_ones, mode='constant', constant_values=1,
    )
    weights = da.pad(
        weights, pads_linear, mode='linear_ramp', end_values=0,
    )
    weights = weights[..., None]

    # multiply data by weights and return
    return da.multiply(block, weights)


def merge_overlaps(block, blocksize):
    """
    """

    p = np.array(blocksize) // 2
    core = [slice(2*x, -2*x) for x in p]
    result = np.copy(block[tuple(core)])

    # faces
    for ax in range(3):
        # the left side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(0, p[ax])
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(0, p[ax])
        result[tuple(slc1)] += block[tuple(slc2)]
        # the right side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(-1*p[ax], None)
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(-1*p[ax], None)
        result[tuple(slc1)] += block[tuple(slc2)]

    # edges
    for edge in product([0, 1], repeat=2):
        for ax in range(3):
            pp = np.delete(p, ax)
            left = [slice(None, pe) for pe in pp]
            right = [slice(-1*pe, None) for pe in pp]
            slc1 = [left[i] if e == 0 else right[i] for i, e in enumerate(edge)]
            slc2 = copy.deepcopy(slc1)
            slc1.insert(ax, slice(None, None))
            slc2.insert(ax, core[ax])
            result[tuple(slc1)] += block[tuple(slc2)]

    # corners
    for corner in product([0, 1], repeat=3):
        left = [slice(None, pe) for pe in p]
        right = [slice(-1*pe, None) for pe in p]
        slc = [left[i] if c == 0 else right[i] for i, c in enumerate(corner)]
        result[tuple(slc)] += block[tuple(slc)]

    return result


def stitch_fields(fields, blocksize):
    """
    """

    # weight block edges
    weighted_fields = da.map_blocks(
        weight_block, fields, blocksize=blocksize, dtype=np.float32,
    )

    # remove block index dimensions
    sh = fields.shape[:3]
    list_of_blocks = [[[[weighted_fields[i,j,k]] for k in range(sh[2])]
                                                 for j in range(sh[1])]
                                                 for i in range(sh[0])]
    aug_fields = da.block(list_of_blocks)

    # merge overlap regions
    overlaps = tuple(blocksize.astype(np.int16) // 2) + (0,)

    return da.map_overlap(
        merge_overlaps, aug_fields, blocksize=blocksize,
        depth=overlaps, boundary=0., trim=False,
        dtype=np.float32, chunks=list(blocksize)+[3,],
    )


