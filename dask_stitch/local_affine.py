import numpy as np
import dask.array as da
from dask.delayed import delayed
from itertools import product


def position_grid(shape, blocksize):
    """
    """

    coords = da.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = da.stack(coords, axis=-1).astype(np.int16)
    return da.rechunk(coords, chunks=tuple(blocksize) + (3,))


def affine_to_field(matrix, grid, displacement=True):
    """
    """

    # apply affine to coordinates
    mm = matrix[:3, :3]
    tt = matrix[:3, -1]
    result = np.einsum('...ij,...j->...i', mm, grid) + tt

    # convert positions to displacements
    if displacement:
        result = result - grid

    # return
    return result


def merge_neighbors(
    grid, affines, overlap,
    displacement,
    block_info=None,
):
    """
    """

    # initialize container for result
    deform = np.zeros_like(grid)

    # get block size, block index, and block grid
    blocksize = np.array(grid.shape[:3])
    block_index = np.array(block_info[0]['chunk-location'][:3])
    block_grid = np.array(block_info[0]['num-chunks'][:3])

    # create weights array
    # determine which faces need linear weighting
    core, pad_ones, pad_linear = [], [], []
    for i in range(3):

        # get core shape and pad sizes
        o = max(0, 2*overlap[i]-1)
        c = blocksize[i] - o + 1
        p_ones, p_linear = [0, 0], [o, o]
        if block_index[i] == 0:
            p_ones[0], p_linear[0] = o, 0
        if block_index[i] == block_grid[i] - 1:
            p_ones[1], p_linear[1] = o, 0
        core.append(c)
        pad_ones.append(tuple(p_ones))
        pad_linear.append(tuple(p_linear))

    # create weights core
    weights = np.ones(core, dtype=np.float32)

    # extend weights
    weights = np.pad(
        weights, pad_ones, mode='constant', constant_values=1,
    )
    weights = np.pad(
        weights, pad_linear, mode='linear_ramp', end_values=0,
    )

    # construct deformation
    # iterate over all neighbors
    for neighbor_offset in product([-1, 0, 1], repeat=3):
        iii = block_index + neighbor_offset
        if np.all(iii >= 0) and np.all(iii < block_grid):

            # determine the block and weight slices
            block_slc, weight_slc = [], []
            for no, o in zip(neighbor_offset, overlap):
                bs, ws = slice(None, None), slice(o, -o)
                if no == -1:
                    bs, ws = slice(0, o), slice(o-1, None, -1)
                elif no == 1:
                    bs, ws = slice(-o, None), slice(-1, -o-1, -1) 
                block_slc.append(bs)
                weight_slc.append(ws)
            block_slc = tuple(block_slc)
            weight_slc = tuple(weight_slc)

            # get vector field
            vec = affine_to_field(
                affines[iii[0], iii[1], iii[2]],
                grid[block_slc],
                displacement,
            )

            # add weighted vector field to result
            deform[block_slc] += vec * weights[weight_slc][..., None]

    # return result
    return deform


def local_affines_to_field(
    shape, spacing, affines,
    blocksize, overlap,
    displacement=True,
):
    """
    """

    # get a coordinate grid
    grid = position_grid(
        np.array(blocksize) * affines.shape[:3], blocksize,
    ) * spacing.astype(np.float32)

    # wrap local_affines as delayed
    affines_d = delayed(affines)

    # map function over blocks
    coords = da.map_blocks(
        merge_neighbors, grid,
        affines=affines_d,
        overlap=overlap,
        displacement=displacement,
        dtype=np.float32,
    )

    # crop to original shape
    coords = coords[:shape[0], :shape[1], :shape[2]]

    # return result
    return coords
        
