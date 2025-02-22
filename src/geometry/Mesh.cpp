#include "Mesh.h"

Mesh::Mesh() {}

Mesh::Mesh(Sphere& s)
{
    std::vector<float3> vertices = s.getVertices();
    std::vector<float3> normals = s.getNormals();
    std::vector<unsigned int> indices = s.getIndices();

    for (size_t i = 0; i < vertices.size(); i++) {
		m_positions.push_back(vertices[i]);
		m_normals.push_back(normals[i]);
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
        m_primitives.push_back(make_uint3(indices[i], indices[i + 1], indices[i + 2]));
    }
}

Mesh::Mesh(Plane& p)
{
    std::vector<float3> vertices = p.getVertices();
    std::vector<float3> normals = p.getNormals();
    std::vector<unsigned int> indices = p.getIndices();

    for (size_t i = 0; i < vertices.size(); i++) {
        m_positions.push_back(vertices[i]);
        m_normals.push_back(normals[i]);
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
        m_primitives.push_back(make_uint3(indices[i], indices[i + 1], indices[i + 2]));
    }
}


Mesh::~Mesh() {}

size_t Mesh::getVertexCount() const
{
    return m_positions.size();
}
