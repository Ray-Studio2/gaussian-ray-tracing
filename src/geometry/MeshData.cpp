#include "MeshData.h"

MeshData::MeshData() {}

float3 MeshData::transform_position(float3 position, glm::mat4 transform)
{
    glm::vec4 glm_position = glm::vec4(position.x, position.y, position.z, 1.0f);
    glm::vec4 transformed_position = transform * glm_position;
	return make_float3(transformed_position.x, transformed_position.y, transformed_position.z);
}

float3 MeshData::transform_normal(float3 normal, glm::mat4 transform)
{
    glm::vec3 glm_normal = glm::vec3(normal.x, normal.y, normal.z);
    glm::vec3 transformed_normal = glm::mat3(transform) * glm_normal;
	return make_float3(transformed_normal.x, transformed_normal.y, transformed_normal.z);
}


std::vector<Vertex> MeshData::transformed_vertices(glm::mat4 transform)
{
	std::vector<Vertex> transformed_vertices;
	for (size_t i = 0; i < m_vertices.size(); i++) {
		Vertex vertex = m_vertices[i];
		vertex.position = transform_position(vertex.position, transform);
		vertex.normal = transform_normal(vertex.normal, transform);
        transformed_vertices.push_back(vertex);
	}
	return transformed_vertices;
}

void MeshData::addMesh(Mesh& m)
{
    std::vector<float3> vertices = m.getVertices();
    std::vector<float3> normals = m.getNormals();
    std::vector<unsigned int> indices = m.getIndices();
    glm::mat4 transform = m.getTransform();

    size_t last_mesh_num = getMeshCount();
    if (last_mesh_num == 0) {
        m_offsets.push_back({ 0, 0 });
    }
    else {
        m_offsets.push_back({ m_vertices.size(), m_primitives.size() });
    }

    for (size_t i = 0; i < vertices.size(); i++) {
        //Vertex v = { transform_position(vertices[i], transform), transform_normal(normals[i], transform) };
        Vertex v = { vertices[i], normals[i] };
		m_vertices.push_back(v);
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
        m_primitives.push_back(make_uint3(indices[i], indices[i + 1], indices[i + 2]));
    }
}

MeshData::~MeshData() {}

size_t MeshData::getMeshCount() const
{
    return m_offsets.size();
}

size_t MeshData::getVertexCount() const
{
	return m_vertices.size();
}

size_t MeshData::getPrimitiveCount() const
{
	return m_primitives.size();
}