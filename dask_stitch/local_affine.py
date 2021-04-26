import numpy as np
import dask.array as da
from dask_stitch.stitch import weight_block, merge_overlaps


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
    )
    grid = grid * spacing.astype(np.float32)
    grid = grid[..., None]  # needed for map_overlap

    # wrap local_affines as dask array
    affines_da = da.from_array(
        affines, chunks=(1, 1, 1, 4, 4),
    )

    # convert to weighted fields
    def affine_to_weighted_field(x, y, block_info=None):
        field = affine_to_field(
            x.squeeze(), y.squeeze(),
            displacement=displacement,
        )
        return weight_block(
            field, blocksize, overlap, block_info,
        )

    # compute affine transforms as displacement fields, lazy dask arrays
    blocksize_with_overlaps = tuple(x+2*y for x, y in zip(blocksize, overlap))
    fields = da.map_overlap(
        affine_to_weighted_field,
        affines_da, grid,
        depth=[0, tuple(overlap)+(0, 0)],
        boundary=0,
        trim=False,
        align_arrays=False,
        dtype=np.float32,
        drop_axis=[4,],
        chunks=blocksize_with_overlaps+(3,),
    )

    # stitch affine position fields
    coords = da.map_overlap(
        merge_overlaps,
        fields,
        overlap=overlap,
        depth=tuple(overlap) + (0,),
        boundary=0.,
        trim=False,
        dtype=np.float32,
        chunks=list(blocksize) + [3,],
    )

    # crop to original shape
    coords = coords[:shape[0], :shape[1], :shape[2]]

    # return result
    return coords

