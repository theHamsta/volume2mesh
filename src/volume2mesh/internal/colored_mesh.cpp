/*
<%
setup_pybind11(cfg)
cfg['linker_args'] = ['-lopenvdb', '-ltbb', '-lHalf', '-lIexMath', '-lIex', '-lIlmThread','-lImath', '-lOpenMeshTools', '-lOpenMeshCore', '-L/home/rzlin/xu29mapu/.local/lib']
cfg['compiler_args'] = ['-std=c++14', '-I/home/rzlin/xu29mapu/.local/include/']
%>
*/
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <string>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include "ColorFilteredMeshDataAdapter.hpp"
#define NUMBER_OF_POINTS_TRIANGLE (3)

namespace py = pybind11;
using OpenMesh_T = OpenMesh::PolyMesh_ArrayKernelT<>;
using OpenTriMesh_T = OpenMesh::TriMesh_ArrayKernelT<>;

// template <typename T>
// using Return_T = std::tuple < py::array_t<T>, std::vector<py::array_t<T>>, std::vector<std::array<int, 3>>>;

template <typename T>
py::tuple meshToDomain( const std::string& filename, const double scaling, const double exteriorBandWidth, const double interiorBandWidth )
{
	using Grid_T = openvdb::Grid<typename openvdb::tree::Tree4<T, 5, 4, 3>::Type >;
	openvdb::initialize();

	auto mesh = std::make_shared<OpenTriMesh_T>();
	OpenMesh::IO::Options opt;
	opt += OpenMesh::IO::Options::FaceColor;
	mesh->request_face_colors();

	if ( !OpenMesh::IO::read_mesh( *mesh, filename, opt ) ) {
		throw std::runtime_error( "Could not read mesh " + filename );
	}

	// Voxelize entire mesh
	openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform( 1 );
	ColorFilteredMeshDataAdapter adapter( mesh, scaling );



	typename Grid_T::Ptr grid = openvdb::tools::meshToVolume<Grid_T, ColorFilteredMeshDataAdapter>( adapter, *linearTransform, exteriorBandWidth, interiorBandWidth );
	auto accessor = grid->getAccessor();
	openvdb::Coord ijk;

	openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

	py::array_t<T> array( {
		boundingBox.dim().z(),
		boundingBox.dim().y(),
		boundingBox.dim().x()
	} );

	// Voxelize uniform tiles (work around)
	grid->tree().voxelizeActiveTiles();

	// Fill with background value
	std::fill( array.mutable_data(), array.mutable_data() + array.size(), static_cast<T>( grid->background() ) );

	auto r = array.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

	// Set sparse voxels
	for ( auto it = grid->beginValueOn(); it; ++it ) {
		auto coord = it.getCoord();
		r( coord.z() - boundingBox.min().z(),
		   coord.y() - boundingBox.min().y(),
		   coord.x() - boundingBox.min().x() ) = it.getValue();
	}

	std::vector<py::array_t<T>> coloredArrays;
	std::vector<std::array<int, 3>> colors;


	// Voxelize colored faces only
	for ( const ColorFilteredMeshDataAdapter::Color_T& color : adapter.colors() ) {
		// Just voxelize 1 to 2 pixel margin around faces of one color
		adapter.setCurrentColor( color );
		grid = openvdb::tools::meshToVolume<Grid_T, ColorFilteredMeshDataAdapter>( adapter, *linearTransform, 1, 1 );
		accessor = grid->getAccessor();

		py::array_t<T> array( {
			boundingBox.dim().z(),
			boundingBox.dim().y(),
			boundingBox.dim().x()
		} );

		// Voxelize uniform tiles (work around)
		grid->tree().voxelizeActiveTiles();

		// Fill with background value
		std::fill( array.mutable_data(), array.mutable_data() + array.size(), static_cast<T>( grid->background() ) );

		auto r = array.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

		// Set sparse voxels
		for ( auto it = grid->beginValueOn(); it; ++it ) {
			auto coord = it.getCoord();
			r( coord.z() - boundingBox.min().z(),
			   coord.y() - boundingBox.min().y(),
			   coord.x() - boundingBox.min().x() ) = it.getValue();
		}

		coloredArrays.push_back( array );
		colors.push_back( {color[0], color[1], color[2]} );
	}

	return py::make_tuple( array, coloredArrays, colors );
}

PYBIND11_MODULE( vdb_domain_from_mesh, m )
{
	m.def( "coloredMeshToDomain", &meshToDomain<float> );
}
