/*
<%
setup_pybind11(cfg)
cfg['linker_args'] = ['-lopenvdb', '-ltbb', '-lHalf', '-lIexMath', '-lIex', '-lIlmThread','-lImath']
cfg['compiler_args'] = ['-std=c++14']
%>
*/
#include <array>
#include <map>
#include <openvdb/openvdb.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// TODO error handling
template< typename T >
py::array_t< T > readVdbGrid(const std::string& filename, const std::string& name,
                             const std::array< int, 3 >& denseShape)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;

    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();
    openvdb::GridBase::Ptr baseGrid;

    if (!name.empty())
    {
        baseGrid = file.readGrid(name);
    }
    else
    {
        auto iter = file.beginName();

        if (iter != file.endName())
        {
            baseGrid = file.readGrid(iter.gridName());
        }
    }

    file.close();

    auto grid = openvdb::gridPtrCast< Grid_T >(baseGrid);

    if (!grid)
    {
        throw std::runtime_error("Could not open VDB Grid");
    }

    openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

    py::array_t< T > array({ std::max(denseShape[0], boundingBox.max().z() + 1),
                             std::max(denseShape[1], boundingBox.max().y() + 1),
                             std::max(denseShape[2], boundingBox.max().x() + 1) });

    // TODO: make work also with uniform tiles
    // openvdb::tools::Dense<T, openvdb::tools::MemoryLayout::LayoutXYZ>
    // denseGrid( boundingBox, array.mutable_data() );
    // openvdb::tools::CopyToDense<typename Grid_T::TreeType, decltype( denseGrid
    // )>( grid->tree(), denseGrid )( boundingBox );

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), static_cast< T >(grid->background()));

    auto r = array.template mutable_unchecked< 3 >(); // Will throw if ndim != 3 or
                                                      // flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord                         = it.getCoord();
        r(coord.z(), coord.y(), coord.x()) = it.getValue();
    }

    return array;
}

template< typename T, int numVectorComponents >
py::array_t< T > readVdbVectorGrid(const std::string& filename, const std::string& name,
                                   const std::array< int, 3 >& denseShape)
{
    using Grid_T = openvdb::Vec3SGrid;

    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();
    openvdb::GridBase::Ptr baseGrid;

    if (!name.empty())
    {
        baseGrid = file.readGrid(name);
    }
    else
    {
        auto iter = file.beginName();

        if (iter != file.endName())
        {
            baseGrid = file.readGrid(iter.gridName());
        }
    }

    file.close();

    auto grid = openvdb::gridPtrCast< Grid_T >(baseGrid);

    if (!grid)
    {
        throw std::runtime_error("Could not open VDB Grid");
    }

    openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

    py::array_t< T > array({ std::max(denseShape[0], boundingBox.max().z() + 1),
                             std::max(denseShape[1], boundingBox.max().y() + 1),
                             std::max(denseShape[2], boundingBox.max().x() + 1), numVectorComponents });

    // TODO: make work also with uniform tiles
    // openvdb::tools::Dense<T, openvdb::tools::MemoryLayout::LayoutXYZ>
    // denseGrid( boundingBox, array.mutable_data() );
    // openvdb::tools::CopyToDense<typename Grid_T::TreeType, decltype( denseGrid
    // )>( grid->tree(), denseGrid )( boundingBox );

    // Voxelize uniform tiles (work around)
    grid->tree().voxelizeActiveTiles();

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), 0);

    auto r = array.template mutable_unchecked< 4 >(); // Will throw if ndim != 3 or
                                                      // flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord = it.getCoord();

        for (int i = 0; i < numVectorComponents; ++i)
        {
            r(coord.z(), coord.y(), coord.x(), i) = it.getValue()[i];
        }
    }

    return array;
}

template< typename T >
std::map< std::string, py::array_t< T > > readVdbGrids(const std::string& filename,
                                                       const std::array< int, 3 >& denseShape)
{
    std::map< std::string, py::array_t< T > > arrays;
    std::vector< std::string > gridNames;

    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();

    for (auto iter = file.beginName(); iter != file.endName(); ++iter)
    {
        gridNames.push_back(iter.gridName());
    }

    file.close();

    for (auto& gridName : gridNames)
    {
        try
        {
            arrays[gridName] = readVdbGrid< T >(filename, gridName, denseShape);
        } catch (...)
        {
            arrays[gridName] = readVdbVectorGrid< T, 3 >(filename, gridName, denseShape);
        }
    }

    return arrays;
}

template< typename T >
py::array_t< T > readVdbGridIntoMemory(const std::string& filename, const std::string& name,
                                       py::array_t< float >& array)
{
    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;

    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();
    openvdb::GridBase::Ptr baseGrid;

    if (!name.empty())
    {
        std::cout << "Reading grid " << name << std::endl;
        baseGrid = file.readGrid(name);
    }
    else
    {
        auto iter = file.beginName();

        if (iter != file.endName())
        {
            baseGrid = file.readGrid(iter.gridName());
        }
    }

    file.close();

    auto grid = openvdb::gridPtrCast< Grid_T >(baseGrid);

    if (!grid)
    {
        throw std::runtime_error("Could not open VDB Grid");
    }

    openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();

    if ((array.shape()[0] < boundingBox.max().z() + 1) || (array.shape()[1] < boundingBox.max().y() + 1) ||
        (array.shape()[2] < boundingBox.max().x() + 1))
    {
        throw std::runtime_error("Array is too small!");
    }

    // Fill with background value
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), static_cast< T >(grid->background()));

    auto r = array.template mutable_unchecked< 3 >(); // Will throw if ndim != 3 or
                                                      // flags.writeable is false

    // Set sparse voxels
    for (auto it = grid->beginValueOn(); it; ++it)
    {
        auto coord                         = it.getCoord();
        r(coord.z(), coord.y(), coord.x()) = it.getValue();
    }

    return array;
}

template< typename T >
void writeVdbGrids(const std::string& filename, std::vector< py::array_t< T > > arrays,
                   const std::vector< std::string >& names, const std::array< double, 3 >& spacing,
                   const float clippingTolerance = 0.f)
{
    openvdb::initialize();

    // TODO: would be more elegant to use std::map
    if (arrays.size() != names.size())
    {
        throw std::runtime_error("You must provide a name for each grid you want to save!");
    }

    using Grid_T = openvdb::Grid< typename openvdb::tree::Tree4< T, 5, 4, 3 >::Type >;

    openvdb::GridPtrVec grids;

    for (uint arrayIdx = 0; arrayIdx < arrays.size(); arrayIdx++)
    {
        auto grid     = Grid_T::create();
        auto accessor = grid->getAccessor();
        openvdb::Coord ijk;

        int &i = ijk[0], &j = ijk[1], &k = ijk[2];

        auto r = arrays[arrayIdx].template unchecked< 3 >();

        for (k = 0; k < r.shape(0); k++)
            for (j = 0; j < r.shape(1); j++)
                for (i = 0; i < r.shape(2); i++)
                    if (r(k, j, i) != 0.f)
                    {
                        accessor.setValue(ijk, r(k, j, i));
                    }

        grid->pruneGrid(clippingTolerance);
        grid->setName(names[arrayIdx]);
        auto scaleTransform = std::make_shared< openvdb::math::Transform >();
        scaleTransform->preScale({ spacing[0], spacing[1], spacing[2] });
        grid->setTransform(scaleTransform);

        grids.push_back(grid);
    }

    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
}

template< typename T >
void writeVdbGrid(const std::string& filename, py::array_t< T > array, std::string& name,
                  const std::array< double, 3 >& spacing, const float clippingTolerance = 0.f)
{
    std::vector< py::array_t< T > > arrayVector{ array };
    std::vector< std::string > nameVector{ name };
    writeVdbGrids(filename, arrayVector, nameVector, spacing, clippingTolerance);
}

PYBIND11_MODULE(vdb_io, m)
{
    m.def("writeFloatVdbGrid", &writeVdbGrid< float >);
    m.def("writeFloatVdbGrid", &writeVdbGrids< float >);
    m.def("readFloatVdbGrid", &readVdbGridIntoMemory< float >);
    m.def("readFloatVdbGrid", &readVdbGrid< float >);
    m.def("readFloatVdbGrid", &readVdbGrids< float >);
    m.def("readIntVdbGrid", &readVdbGrid< int >);
    m.def("readIntVdbGrid", &readVdbGrids< int >);
}
