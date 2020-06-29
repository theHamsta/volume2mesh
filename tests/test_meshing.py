# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from os.path import dirname, join

import numpy as np

import volume2mesh

num_samples = 100000


def test_mesh2volume():
    bunny_file = join(dirname(__file__), 'data', 'bunny.obj')
    volume = volume2mesh.mesh2volume(bunny_file, scaling=1.)
    assert volume.any()


def test_roundtrip():
    bunny_file = join(dirname(__file__), 'data', 'bunny.obj')
    volume = volume2mesh.mesh2volume(bunny_file, scaling=1.)
    assert volume.any()
    import pyconrad.autoinit
    pyconrad.imshow(volume)
    volume2mesh.volume2mesh('/tmp/new_bunny.obj', volume, 0.1)


def test_vdb_io():
    random_sparse_grid = np.zeros((100, 120, 131), np.float32)

    for _ in range(num_samples):
        ijk = [np.random.randint(0, i-1) for i in random_sparse_grid.shape]
        random_value = np.random.randn(1)
        random_sparse_grid[tuple(ijk)] = random_value
    random_sparse_grid = np.ascontiguousarray(random_sparse_grid)

    volume2mesh.write_vdb('/tmp/foox.vdb', random_sparse_grid,
                          'random_sparse_grid')

    read_grid = volume2mesh.read_vdb(
        '/tmp/foox.vdb', 'random_sparse_grid', random_sparse_grid.shape)

    assert np.allclose(read_grid, random_sparse_grid), 'Grids not equal'


def test_multichannel_grid():
    grid_shape = (100, 120, 131)
    grids = []
    for _ in range(4):
        random_sparse_grid = np.zeros(grid_shape, np.float32)

        for _ in range(num_samples):
            ijk = [np.random.randint(0, i-1) for i in random_sparse_grid.shape]
            random_value = np.random.randn(1)
            random_sparse_grid[tuple(ijk)] = random_value
        random_sparse_grid = np.ascontiguousarray(random_sparse_grid)
        grids.append(random_sparse_grid)

    volume2mesh.write_vdb('/tmp/multichannel.vdb', grids,
                          ['grid%i' % i for i in range(len(grids))])

    read_grids = volume2mesh.read_vdb(
        '/tmp/multichannel.vdb', dense_shape=grid_shape)

    for i in range(len(grids)):
        assert np.allclose(
            grids[i], read_grids['grid%i' % i]), 'Grids not equal (index %i)' % i


def test_save_mesh_from_volume():

    volume = np.zeros((100, 120, 131), np.float32)
    volume[20:40, 30:40, 40:50] = 1

    volume2mesh.volume2mesh('/tmp/foo.obj', volume, threshold=0.5)


def read_volume_from_mesh_test():
    array = volume2mesh.mesh2volume(join(dirname(__file__), "data", "cube.stl"), 100)
    assert array is not None


def read_domain_from_mesh_test():

    array, colored_faces, colors = volume2mesh.mesh_to_domain(
        join(dirname(__file__), "data", "material_cub.obj"), 100)

    for i, c in enumerate(colored_faces):
        assert c.any()
        assert colors[i]
    assert array is not None
