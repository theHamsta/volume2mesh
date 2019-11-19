# -*- coding: utf-8 -*-
import cppimport
import numpy as np
from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

_vdb_io = None
_vdb_meshing = None
_colored_mesh = None


def read_vdb(file, grid_name='', dense_shape=[0]*3, array=None):
    global _vdb_io
    if not _vdb_io:
        _vdb_io = cppimport.imp('volume2mesh.internal.vdb_io')

    # dense_shape = list(dense_shape)
    if grid_name == '':
        try:
            rtn = _vdb_io.readFloatVdbGrid(file, dense_shape)
        except Exception:
            rtn = _vdb_io.readIntVdbGrid(file, dense_shape)
        # if len(rtn) == 1:
        #     return next(iter(rtn.values()))
    elif array is None:
        rtn = _vdb_io.readFloatVdbGrid(file, grid_name, dense_shape)
    else:
        rtn = _vdb_io.readFloatVdbGrid(file, array, grid_name, dense_shape)

    return rtn


def write_vdb(file, array, grid_name, spacing=[1., 1., 1.], origin=[0., 0., 0.], quantization_tolerance=0.):
    global _vdb_io
    if not _vdb_io:
        _vdb_io = cppimport.imp('volume2mesh.internal.vdb_io')
    array = np.ascontiguousarray(array, np.float32)
    _vdb_io.writeFloatVdbGrid(file, array, grid_name, spacing, origin,  quantization_tolerance)


def volume2mesh(file,
                volume,
                isovalue=0.,
                adaptivity=0.,
                spacing=[1., 1., 1.],
                origin=[1., 1., 1.],
                binary_file=True,
                only_write_biggest_components=False,
                max_component_count=1):
    global _vdb_meshing
    if not _vdb_meshing:
        _vdb_meshing = cppimport.imp('volume2mesh.internal.vdb_meshing')
    _vdb_meshing.writeMeshFromVolume(
        file,
        volume,
        isovalue,
        adaptivity,
        spacing,
        origin,
        binary_file,
        only_write_biggest_components,
        max_component_count)


def mesh2volume(mesh_file, scaling, exterior_band=1, interior_band=1000):
    global _vdb_meshing
    if not _vdb_meshing:
        _vdb_meshing = cppimport.imp('volume2mesh.internal.vdb_meshing')
    return _vdb_meshing.meshToVolume(mesh_file, scaling, exterior_band, interior_band)


def mesh_to_signed_distance_field(mesh_file, scaling, exterior_band=1, interior_band=1000):
    global _vdb_meshing
    if not _vdb_meshing:
        _vdb_meshing = cppimport.imp('volume2mesh.internal.vdb_meshing')
    return _vdb_meshing.meshToSignedDistanceField(mesh_file, scaling, exterior_band, interior_band)


def colored_mesh_to_volumes(mesh_file, scaling, exterior_band=1, interior_band=1000):
    global _colored_mesh
    if not _colored_mesh:
        _colored_mesh = cppimport.imp('volume2mesh.internal.colored_mesh')
    return _colored_mesh(mesh_file, scaling, exterior_band, interior_band)
