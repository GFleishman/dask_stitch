import numpy as np
import dask.array as da
from dask_stitch.stitch import stitch_blocks


def position_grid(shape, blocksize):
    """
    """

    coords = da.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = da.stack(coords, axis=-1).astype(np.int16)
    return da.rechunk(coords, chunks=tuple(blocksize) + (3,))


def affine_to_grid(matrix, grid):
    """
    """

    # reformat matrix, keep track of trimmed dimensions
    ndims = len(matrix.shape)
    matrix = matrix.astype(np.float32).squeeze()
    lost_dims = ndims - len(matrix.shape)

    # apply affine to coordinates
    mm = matrix[:3, :3]
    tt = matrix[:3, -1]
    result = da.einsum('...ij,...j->...i', mm, grid) + tt

    # convert positions to displacements
    result = result - grid

    # restore trimmed dimensions
    if lost_dims > 0:
        result = result.reshape((1,)*lost_dims + result.shape)
    return result


def local_affine_to_displacement(shape, spacing, affines, blocksize):
    """
    """

    # define some helpful variables
    overlaps = list(blocksize // 2)
    nblocks = affines.shape[:3]

    # adjust affines for block origins
    for i in range(np.prod(nblocks)):
        x, y, z = np.unravel_index(i, nblocks)
        origin = np.maximum(
            np.array(blocksize) * [x, y, z] - overlaps, 0,
        )
        origin = origin * spacing
        tl, tr = np.eye(4), np.eye(4)
        a, tl[:3, -1], tr[:3, -1] = affines[x, y, z], origin, -origin
        affines[x, y, z] = np.matmul(tl, np.matmul(a, tr))

    # get a coordinate grid
    grid = position_grid(
        np.array(blocksize) * nblocks, blocksize,
    )
    grid = grid * spacing.astype(np.float32)
    grid = grid[..., None]  # needed for map_overlap

    # wrap local_affines as dask array
    affines_da = da.from_array(
        affines, chunks=(1, 1, 1, 4, 4),
    )

    # strip dummy axis off grid
    def wrapped_affine_to_grid(x, y):
        y = y.squeeze()
        return affine_to_grid(x, y)

    # compute affine transforms as displacement fields, lazy dask arrays
    blocksize_with_overlaps = tuple(x+2*y for x, y in zip(blocksize, overlaps))
    coords = da.map_overlap(
        wrapped_affine_to_grid, affines_da, grid,
        depth=[0, tuple(overlaps)+(0, 0)],
        boundary=0,
        trim=False,
        align_arrays=False,
        dtype=np.float32,
        new_axis=[5,6],
        chunks=(1,1,1,) + blocksize_with_overlaps + (3,),
    )

    # stitch affine position fields
    coords = stitch_blocks(coords, blocksize)

    # crop to original shape
    coords = coords[:shape[0], :shape[1], :shape[2]]

    # return result
    return coords


