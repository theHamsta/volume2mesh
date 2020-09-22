/*
<%
setup_pybind11(cfg)
cfg['linker_args'] = ['-lopenvdb',
 '-ltbb',
 '-lHalf',
 '-lIexMath',
 '-lIex',
 '-lIlmThread',
'-lImath',
 '-lOpenMeshTools',
'-lOpenMeshCore']
cfg['compiler_args'] = ['-std=c++14']
%>
*/
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <algorithm>
#include <array>
#include <memory>
#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "MeshDataAdapter.hpp"
#define NUMBER_OF_POINTS_TRIANGLE (3)

namespace py        = pybind11;
using OpenMesh_T    = OpenMesh::PolyMesh_ArrayKernelT<>;
using OpenTriMesh_T = OpenMesh::TriMesh_ArrayKernelT<>;
using namespace pybind11::literals;

template< class Mesh_T >
void selectBiggestComponents(Mesh_T* _mesh, int maxComponentCount);

template< class Mesh_T >
void deletedUnselectedFaces(Mesh_T& mesh)
{
    assert(mesh.has_vertex_status() && "Mesh needs to have vertex status");
    assert(mesh.has_face_status() && "Mesh needs to have face status");

    for (typename Mesh_T::FaceIter fIter = mesh.faces_begin(); fIter != mesh.faces_end(); ++fIter)
    {
        if (!mesh.status(*fIter).selected())
        {
            mesh.delete_face(*fIter, true);
        }
    }
    mesh.garbage_collection();
}

template< typename T >
void writeMeshFromVolume(const std::string& filename, py::array_t< T > array, const float isovalue = 0.f,
                         const float adaptivity = 0.f, std::array< double, 3 > spacing = { 1., 1., 1. },
                         std::array< double, 3 > origin = { 0., 0., 0. }, bool writeBinaryMeshFile = true,
                         bool onlyWriteBiggestComponents = false, int maxComponentCount = 1)
{
    openvdb::initialize();

    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;

    auto grid           = Grid_T::create();
    auto scaleTransform = std::make_shared< openvdb::math::Transform >();
    scaleTransform->preScale({ 1. * spacing[0], 1. * spacing[1], 1. * spacing[2] });
    scaleTransform->postTranslate({ +origin[0], +origin[1], +origin[2] });
    grid->setTransform(scaleTransform);

    auto accessor = grid->getAccessor();
    openvdb::Coord ijk;

    int &i = ijk[0], &j = ijk[1], &k = ijk[2];

    auto r = array.template unchecked< 3 >();

    for (k = 0; k < r.shape(0); k++)
        for (j = 0; j < r.shape(1); j++)
            for (i = 0; i < r.shape(2); i++)
                if (r(k, j, i) != 0.f)
                {
                    accessor.setValue(ijk, -r(k, j, i)); // - is to fix normals (positive should be inside)
                }

    grid->pruneGrid(0.f);

    std::vector< openvdb::Vec3s > points;
    std::vector< openvdb::Vec3I > triangles;
    std::vector< openvdb::Vec4I > quads;
    openvdb::tools::volumeToMesh< Grid_T >(*grid, points, triangles, quads, isovalue, adaptivity);

    OpenMesh_T mesh;

    for (int i = 0; i < static_cast< int >(points.size()); i++)
    {
        mesh.add_vertex({ points[i][0], points[i][1], points[i][2] });
    }

    for (int i = 0; i < static_cast< int >(quads.size()); i++)
    {
        auto point0 = mesh.vertex_handle(quads[i][0]);
        auto point1 = mesh.vertex_handle(quads[i][1]);
        auto point2 = mesh.vertex_handle(quads[i][2]);
        auto point3 = mesh.vertex_handle(quads[i][3]);
        std::vector< OpenMesh_T::VertexHandle > face_vertices{ point0, point1, point2, point3 };
        mesh.add_face(face_vertices);
    }

    if (onlyWriteBiggestComponents)
    {
        selectBiggestComponents(&mesh, maxComponentCount);
        deletedUnselectedFaces(mesh);
    }
    OpenMesh::IO::Options options(writeBinaryMeshFile ? OpenMesh::IO::Options::Binary : OpenMesh::IO::Options::Default);
    OpenMesh::IO::write_mesh(mesh, filename, options);
}

template< class Mesh_T >
void selectBiggestComponents(Mesh_T* _mesh, int maxComponentCount)
{
    OpenMesh::FPropHandleT< bool > visited;
    if (!_mesh->get_property_handle(visited, "visited"))
    {
        _mesh->add_property(visited, "visited");
    }
    _mesh->request_face_status();
    _mesh->request_vertex_status();

    std::vector< std::vector< typename Mesh_T::FaceHandle > > facesByComponent;

    [[maybe_unused]] int numComponents = 0;
    for (typename Mesh_T::FaceIter fIter = _mesh->faces_begin();

         fIter != _mesh->faces_end(); ++fIter)
    {
        if (_mesh->property(visited, *fIter))
            continue;

        // It is a vertex, which is not visited => new component
        std::vector< typename Mesh_T::FaceHandle > componentFaces;
        componentFaces.push_back(*fIter);
        numComponents++;

        for (std::size_t i = 0; i < componentFaces.size(); ++i)
        {
            // add all not visited neightbors
            for (typename Mesh_T::FaceFaceIter ffIter = _mesh->ff_begin(componentFaces[i]); ffIter.is_valid(); ++ffIter)
            {
                if (!ffIter->is_valid())
                    std::cout << "handleId: " << *ffIter << std::endl;
                if (ffIter->is_valid() && !_mesh->property(visited, *ffIter))
                {
                    _mesh->property(visited, *ffIter) = true;
                    componentFaces.push_back(*ffIter);
                }
            }
        }

        facesByComponent.push_back(componentFaces);
    }
    std::sort(facesByComponent.begin(), facesByComponent.end(),
              [](const auto& x, const auto& y) { return x.size() > y.size(); });
    _mesh->remove_property(visited);

    // std::cout << "Mesh has " << numComponents << " components"<< std::endl;

    for (int i = 0; i < maxComponentCount && i < static_cast< int >(facesByComponent.size()); ++i)
    {
        for (typename std::vector< typename Mesh_T::FaceHandle >::iterator iter = facesByComponent[i].begin();
             iter != facesByComponent[i].end(); ++iter)
        {
            _mesh->status(*iter).set_selected(true);
        }
    }
}

template< typename T >
py::array_t< T > meshToVolume(const std::string& filename, const std::array< double, 3 > spacing,
                              const double exteriorBandWidth, const double interiorBandWidth)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;
    openvdb::initialize();

    auto mesh = std::make_shared< OpenTriMesh_T >();

    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        throw std::runtime_error("Could not read mesh " + filename);
    }

    openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform(1);
    MeshDataAdapter adapter(mesh, { 0., 0., 0. }, spacing);

    typename Grid_T::Ptr grid = openvdb::tools::meshToVolume< Grid_T, MeshDataAdapter >(
        adapter, *linearTransform, exteriorBandWidth, interiorBandWidth);
    auto accessor = grid->getAccessor();
    openvdb::Coord ijk;

    openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

    py::array_t< T > array({ boundingBox.dim().z(), boundingBox.dim().y(), boundingBox.dim().x() });

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), static_cast< T >(0.));

    auto r = array.template mutable_unchecked< 3 >(); // Will throw if ndim != 3 or flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord = it.getCoord();
        r(coord.z() - boundingBox.min().z(), coord.y() - boundingBox.min().y(), coord.x() - boundingBox.min().x()) =
            1 - it.getValue(); // - is for fixing the sign, (+) should be inside
    }

    return array;
}

template< typename T >
py::array_t< T > meshToVolumeKnownDimensions(const std::string& filename, const std::array< double, 3 > origin,
                                             const std::array< double, 3 > spacing,
                                             const std::array< int, 3 > volumeDimensions,
                                             const double exteriorBandWidth, const double interiorBandWidth)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;
    openvdb::initialize();

    auto mesh = std::make_shared< OpenTriMesh_T >();

    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        throw std::runtime_error("Could not read mesh " + filename);
    }

    openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform();
    linearTransform->postScale({ spacing[0], spacing[1], spacing[2] });
    linearTransform->postTranslate({ origin[0], origin[1], origin[2] });

    MeshDataAdapter adapter(mesh, origin, spacing);

    typename Grid_T::Ptr grid = openvdb::tools::meshToVolume< Grid_T, MeshDataAdapter >(
        adapter, *linearTransform, exteriorBandWidth, interiorBandWidth);
    auto accessor = grid->getAccessor();
    openvdb::Coord ijk;
    openvdb::Coord minCoordinate{ 0, 0, 0 };
    openvdb::Coord maxCoordinate{ volumeDimensions[0], volumeDimensions[1], volumeDimensions[2] };

    py::array_t< T > array({ volumeDimensions[2], volumeDimensions[1], volumeDimensions[0] });

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), static_cast< T >(0.));

    auto r = array.template mutable_unchecked< 3 >(); // Will throw if ndim != 3 or flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord                         = it.getCoord();
        r(coord.z(), coord.y(), coord.x()) = 1 - it.getValue(); // - is for fixing the sign, (+) should be inside
    }

    return array;
}

template< typename T >
py::array_t< T > meshToSignedDistanceField(const std::string& filename, const double scaling,
                                           const double exteriorBandWidth, const double interiorBandWidth)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;
    openvdb::initialize();

    auto mesh = std::make_shared< OpenTriMesh_T >();

    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        throw std::runtime_error("Could not read mesh " + filename);
    }

    openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform(1);
    std::vector< openvdb::Vec3s > points;
    std::vector< openvdb::Vec3I > triangles;
    std::vector< openvdb::Vec4I > quads;

    points.reserve(mesh->n_vertices());
    triangles.reserve(mesh->n_faces());

    for (auto it = mesh->vertices_begin(); it != mesh->vertices_end(); ++it)
    {
        auto point = mesh->point(*it);
        points.push_back({ point[0], point[1], point[2] });
    }

    for (auto it = mesh->faces_begin(); it != mesh->faces_end(); ++it)
    {
        auto face_vertex_iterator = mesh->cfv_iter(*it);

        std::array< unsigned int, 3 > trianglePoints;
        for (size_t i = 0; i < NUMBER_OF_POINTS_TRIANGLE; i++)
        {
            trianglePoints[i] = static_cast< uint >(face_vertex_iterator->idx());
        }
        triangles.push_back({ trianglePoints[0], trianglePoints[1], trianglePoints[2] });
    }

    typename Grid_T::Ptr grid = openvdb::tools::meshToSignedDistanceField< Grid_T >(
        *linearTransform, points, triangles, quads, exteriorBandWidth, interiorBandWidth);
    openvdb::Coord ijk;

    openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

    py::array_t< T > array({ boundingBox.dim().z(), boundingBox.dim().y(), boundingBox.dim().x() });

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), static_cast< T >(1 - grid->background()));

    auto r = array.template mutable_unchecked< 3 >(); // Will throw if ndim != 3 or flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord = it.getCoord();
        r(coord.z() - boundingBox.min().z(), coord.y() - boundingBox.min().y(), coord.x() - boundingBox.min().x()) =
            -it.getValue(); // -x is for fixing the sign, positive should be inside
    }

    return array;
}

template< typename T >
py::array_t< T > calcVertexDataFromScalarField(const std::string& filename, const py::array_t< T > scalarField,
                                               const std::array< double, 3 > origin,
                                               const std::array< double, 3 > spacing, const double exteriorBandWidth,
                                               const double interiorBandWidth)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;
    openvdb::initialize();

    auto mesh = std::make_shared< OpenTriMesh_T >();

    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        throw std::runtime_error("Could not read mesh " + filename);
    }

    openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform();
    linearTransform->postScale({ spacing[0], spacing[1], spacing[2] });
    linearTransform->postTranslate({ origin[0], origin[1], origin[2] });

    MeshDataAdapter adapter(mesh, origin, spacing);

    typename Grid_T::Ptr grid = openvdb::tools::meshToVolume< Grid_T, MeshDataAdapter >(
        adapter, *linearTransform, exteriorBandWidth, interiorBandWidth);
    auto accessor = grid->getAccessor();
    openvdb::Coord ijk;
    openvdb::Coord minCoordinate{ 0, 0, 0 };
    openvdb::Coord maxCoordinate{ static_cast< int32_t >(scalarField.shape(2)),
                                  static_cast< int32_t >(scalarField.shape(1)),
                                  static_cast< int32_t >(scalarField.shape(0)) };

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    auto scalarFieldAccessor =
        scalarField.template unchecked< 3 >(); // Will throw if ndim != 3 or flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord = it.getCoord();
        accessor.setValue(coord, scalarFieldAccessor(coord.z(), coord.y(), coord.x()));
    }

    py::array_t< T > array({ mesh->n_vertices() });

    auto v = mesh->vertices_begin();
    for (size_t i = 0; v != mesh->vertices_end(); ++i, ++v)
    {
        auto p = mesh->point(*v);

        const openvdb::Vec3R coordinate{ (p[0] - origin[0]) / spacing[0], (p[1] - origin[1]) / spacing[1],
                                         (p[2] - origin[2]) / spacing[2] };
        array.mutable_data()[i] = static_cast< T >(openvdb::tools::BoxSampler::sample(grid->tree(), coordinate));
    }

    return array;
}

template< typename T >
std::tuple< py::array_t< T >, py::array_t< T > >
    calcFaceDataFromScalarField(const std::string& filename, const py::array_t< T > scalarField,
                                const std::array< double, 3 > origin, const std::array< double, 3 > spacing,
                                const double exteriorBandWidth, const double interiorBandWidth, bool calcArea)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;
    openvdb::initialize();

    auto mesh = std::make_shared< OpenTriMesh_T >();

    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        throw std::runtime_error("Could not read mesh " + filename);
    }

    openvdb::math::Transform::Ptr linearTransform = openvdb::math::Transform::createLinearTransform();
    linearTransform->postScale({ spacing[0], spacing[1], spacing[2] });
    linearTransform->postTranslate({ origin[0], origin[1], origin[2] });

    MeshDataAdapter adapter(mesh, origin, spacing);

    typename Grid_T::Ptr grid = openvdb::tools::meshToVolume< Grid_T, MeshDataAdapter >(
        adapter, *linearTransform, exteriorBandWidth, interiorBandWidth);
    auto accessor = grid->getAccessor();
    openvdb::Coord ijk;
    openvdb::Coord minCoordinate{ 0, 0, 0 };
    openvdb::Coord maxCoordinate{ static_cast< int32_t >(scalarField.shape(2)),
                                  static_cast< int32_t >(scalarField.shape(1)),
                                  static_cast< int32_t >(scalarField.shape(0)) };

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    auto scalarFieldAccessor =
        scalarField.template unchecked< 3 >(); // Will throw if ndim != 3 or flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord = it.getCoord();
        accessor.setValue(coord, scalarFieldAccessor(coord.z(), coord.y(), coord.x()));
    }

    py::array_t< T > array({ mesh->n_faces() });
    py::array_t< T > area_array({ mesh->n_faces() });

    auto f = mesh->faces_begin();
    for (size_t i = 0; f != mesh->faces_end(); ++i, ++f)
    {
        auto fv_iter = mesh->fv_iter(*f);
        auto P       = mesh->point(*fv_iter++);
        auto R       = mesh->point(*fv_iter++);
        auto Q       = mesh->point(*fv_iter++);

        if (calcArea)
        {
            T area                       = static_cast< T >(((Q - P) % (R - P)).norm() * T(0.5));
            area_array.mutable_data()[i] = area;
        }

        const openvdb::Vec3R coordinate{ ((P[0] + R[0] + Q[0]) * T(1 / 3.) - origin[0]) / spacing[0],
                                         ((P[1] + R[1] + Q[1]) * T(1 / 3.) - origin[1]) / spacing[1],
                                         ((P[2] + R[2] + Q[2]) * T(1 / 3.) - origin[2]) / spacing[2] };
        array.mutable_data()[i] = static_cast< T >(openvdb::tools::BoxSampler::sample(grid->tree(), coordinate));
    }

    return { array, area_array };
}

PYBIND11_MODULE(vdb_meshing, m)
{
    m.def("writeMeshFromVolume", &writeMeshFromVolume< float >, "filename"_a, "array"_a, "threshold"_a = 0.f,
          "adaptivity"_a = 0.f, "spacing"_a = std::array< double, 3 >{ 1., 1., 1. },
          "origin"_a = std::array< double, 3 >{ 0., 0., 0. }, "writeBinaryMeshFile"_a = true,
          "onlyWriteBiggestComponents"_a = false, "maxComponentCount"_a = 1);
    m.def("meshToVolume", &meshToVolume< float >);
    m.def("meshToVolumeKnownDimensions", &meshToVolumeKnownDimensions< float >);
    m.def("meshToSignedDistanceField", &meshToVolume< float >);
    m.def("calcVertexDataFromScalarField", &calcVertexDataFromScalarField< float >, "mesh_file"_a, "scalarField"_a,
          "origin"_a, "spacing"_a, "exteriorBandWidth"_a, "interiorBandWidth"_a);
    m.def("calcFaceDataFromScalarField", &calcFaceDataFromScalarField< float >, "mesh_file"_a, "scalarField"_a,
          "origin"_a, "spacing"_a, "exteriorBandWidth"_a, "interiorBandWidth"_a, "calcArea"_a);
}
