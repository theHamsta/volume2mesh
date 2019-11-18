#pragma once

#include <openvdb/openvdb.h>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <memory>
#include <stdexcept>
#include <set>
#include <optional>

#define NUMBER_OF_POINTS_TRIANGLE (3)

class ColorFilteredMeshDataAdapter
{
	public:
		using OpenTriMesh_T = OpenMesh::TriMesh_ArrayKernelT<>;
		using Color_T = OpenMesh::Vec3uc;

		ColorFilteredMeshDataAdapter( const std::shared_ptr<OpenTriMesh_T> mesh, const double scaling )
			: m_mesh( mesh )
			, m_scaling( scaling )
		{
			if ( !m_mesh->has_face_colors() ) {
				throw std::invalid_argument( "Mesh has no colors! Please use openmesh::IO::FaceColor or check whether your mesh has any face color information" );
			}

			sortFacesByColor();
		};


		size_t polygonCount() const { return m_currentColor ? m_facesByColor.at( *m_currentColor ).size() : m_mesh->n_faces(); }
		size_t pointCount() const { return m_currentColor ? m_verticesByColor.at( *m_currentColor ).size() : m_mesh->n_vertices(); }        // Total number of points
		size_t vertexCount( size_t n ) const {return NUMBER_OF_POINTS_TRIANGLE; } // Vertex count for polygon n
		// Return position pos in local grid index space for polygon n and vertex v
		void getIndexSpacePoint( size_t n, size_t v, openvdb::Vec3d& pos ) const
		{
			const auto face_handle = [&]() {
				if ( m_currentColor ) {
					return m_facesByColor.at( *m_currentColor )[n];
				} else {
					return m_mesh->face_handle( n );
				}
			}();

			auto face_vertex_iterator = m_mesh->fv_iter( face_handle ) ;

			for ( size_t i = 0; i < v; i++ ) {
				face_vertex_iterator++;
			}

			const auto point =  m_mesh->point( *face_vertex_iterator ) ;
			pos[0] = m_scaling * point[0];
			pos[1] = m_scaling * point[1];
			pos[2] = m_scaling * point[2];
		}

		void setCurrentColor( const std::experimental::optional<Color_T>& color ) { m_currentColor = color; }
		const std::experimental::optional<Color_T> currentColor() const { return m_currentColor; }
		const std::set<Color_T>& colors() const { return m_colors; }

	private:
		void sortFacesByColor()
		{

			for ( auto faceIt = m_mesh->faces_begin(); faceIt != m_mesh->faces_end(); ++faceIt ) {

				Color_T faceColor = m_mesh->color( *faceIt );
				m_colors.insert( faceColor );
				m_facesByColor[faceColor].push_back( *faceIt );

				auto fv_iter = m_mesh->cfv_iter( *faceIt );

				for ( int i = 0; i < NUMBER_OF_POINTS_TRIANGLE; ++i ) {
					m_verticesByColor[faceColor].insert( *fv_iter );
					fv_iter++;
				}
			}
		}

		std::shared_ptr<OpenTriMesh_T> m_mesh;
		double m_scaling;
		std::map<Color_T, std::vector<OpenMesh::FaceHandle>> m_facesByColor;
		std::map<Color_T, std::set<OpenMesh::VertexHandle>> m_verticesByColor;
		std::set<OpenMesh::Vec3uc> m_colors;
		std::experimental::optional<Color_T> m_currentColor = std::experimental::nullopt;
};