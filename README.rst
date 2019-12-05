.. image:: https://badge.fury.io/py/volume2mesh.svg
   :target: https://badge.fury.io/py/volume2mesh
   :alt: PyPI version


.. image:: https://travis-ci.org/theHamsta/volume2mesh.svg?branch=master
   :target: https://travis-ci.org/theHamsta/volume2mesh
   :alt: Build Status

===========
volume2mesh
===========

Voxelize meshes to volumes. Save meshes from volumes. 

This package provides to functions ``volume2mesh`` and ``mesh2volume`` to convert between NumPy volume arrays and
meshes:

.. code:: python

   file = '/tmp/my_mesh_file.obj'
   volume = np.zeros((100, 120, 131), np.float32)
   volume[20:40, 30:40, 40:50] = 1
    
   volume2mesh(file,
              volume,
              threshold=0.5,
              adaptivity=0.,
              spacing=[1., 1., 1.],
              origin=[0., 0., 0.],
              binary_file=True,
              only_write_biggest_components=False,
              max_component_count=1)

And to voxelize meshes:

.. code:: python
 
   bunny_file = '~/my_bunnyfile.stl'
   volume = volume2mesh.mesh2volume(bunny_file, scaling=1.)

Installation
------------

You need to have `OpenMesh <https://www.openmesh.org/>`_ linkable via ``-lOpenMeshCore`` and `OpenVDB <openvdb.org>`_ 
installed.

On on Ubuntu, you can do this by these commands (see our `Travis Script <https://github.com/theHamsta/volume2mesh/blob/master/.travis.yml>`_):

.. code:: bash

   # Install OpenVDB
   sudo apt-get install -y libopenvdb-dev build-essential libboost-all-dev libtbb-dev

   # Install OpenMesh
   git clone https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git || echo "hi"
   cd OpenMesh
   git pull 
   mkdir -p release
   cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -- -j4
   sudo make install

   # Install this package
   pip3 install volume2mesh --user
   # Or if you cloned this repo
   pip3 install -e . --user



