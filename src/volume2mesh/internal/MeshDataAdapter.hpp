#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <memory>
#include <openvdb/openvdb.h>
#include <stdexcept>

#define NUMBER_OF_POINTS_TRIANGLE (3)

class MeshDataAdapter
{
  public:
    using OpenTriMesh_T = OpenMesh::TriMesh_ArrayKernelT<>;

    MeshDataAdapter(const std::shared_ptr< OpenTriMesh_T > mesh, const double scaling)
        : m_mesh(mesh), m_scaling(scaling){};

    size_t polygonCount() const { return m_mesh->n_faces(); }
    size_t pointCount() const { return m_mesh->n_vertices(); }               // Total number of points
    size_t vertexCount(size_t n) const { return NUMBER_OF_POINTS_TRIANGLE; } // Vertex count for polygon n
    // Return position pos in local grid index space for polygon n and vertex v
    void getIndexSpacePoint(size_t n, size_t v, openvdb::Vec3d& pos) const
    {
        const auto face_handle    = m_mesh->face_handle(n);
        auto face_vertex_iterator = m_mesh->fv_iter(face_handle);

        for (size_t i = 0; i < v; i++)
        {
            face_vertex_iterator++;
        }

        const auto point = m_mesh->point(*face_vertex_iterator);
        pos[0]           = m_scaling * point[0];
        pos[1]           = m_scaling * point[1];
        pos[2]           = m_scaling * point[2];
    }

  private:
    std::shared_ptr< OpenTriMesh_T > m_mesh;
    double m_scaling;
};
