#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "Primitives.h"

Primitive Primitives::createPlane(float3 position)
{
	Primitive p;

	unsigned int tessU = 1;
	unsigned int tessV = 1;

	const float width = 0.3f;
	const float height = 0.5f;

	const float uTile = width / float(tessU);
	const float vTile = height / float(tessV);

	float3 corner = make_float3(-width * 0.5f, -height * 0.5f, 0.0f);
	float3 normal = make_float3(0.0f, 0.0f, 1.0f);

	for (unsigned int j = 0; j <= tessV; ++j)
	{
		const float v = float(j) * vTile;

		for (unsigned int i = 0; i <= tessU; ++i)
		{
			const float u = float(i) * uTile;

			float3 vertex = corner + make_float3(u, v, 0.0f);

			p.vertices.push_back(vertex);
			p.normals.push_back(normal);
		}
	}

	const unsigned int stride = tessU + 1;
	for (unsigned int j = 0; j < tessV; ++j)
	{
		for (unsigned int i = 0; i < tessU; ++i)
		{
			p.indices.push_back(j * stride + i);
			p.indices.push_back(j * stride + i + 1);
			p.indices.push_back((j + 1) * stride + i + 1);

			p.indices.push_back((j + 1) * stride + i + 1);
			p.indices.push_back((j + 1) * stride + i);
			p.indices.push_back(j * stride + i);
		}
	}

	p.index = numberOfPlane++;
	p.type = "Plane";
    p.instanceIndex = numberOfMesh++;
	p.vertex_count = p.vertices.size();
	p.transform = getInitialTransform(position);

	m_primitives.push_back(p);

	return p;
}

Primitive  Primitives::createSphere(float3 position)
{
    Primitive p;

    // TODO: Hardcoded..
    unsigned int tessU = 180;
    unsigned int tessV = 90;
    float        radius = 0.3f;
    float        maxTheta = M_PIf;

    p.vertices.reserve((tessU + 1) * tessV);
    p.indices.reserve(6 * tessU * (tessV - 1));

    float phi_step = 2.0f * M_PIf / (float)tessU;
    float theta_step = maxTheta / (float)(tessV - 1);

    // Latitudinal rings.
    // Starting at the south pole going upwards on the y-axis.
    for (unsigned int latitude = 0; latitude < tessV; ++latitude) // theta angle
    {
        float theta = (float)latitude * theta_step;
        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]

        // Generate vertices along the latitudinal rings.
        // On each latitude there are tessU + 1 vertices.
        // The last one and the first one are on identical positions, but have different texture coordinates!
        // Note that each second triangle connected to the two poles has zero area if the sphere is closed.
        // But since this is also used for open spheres (maxTheta < 1.0f) this is required.
        for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
        {
            float phi = (float)longitude * phi_step;
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]

            // Unit sphere coordinates are the normals.
            float3 normal = make_float3(cosPhi * sinTheta,
                cosTheta,           // -y to start at the south pole.
                sinPhi * sinTheta);
            float3 vertex;
            vertex = normal * radius;
            p.vertices.push_back(vertex);
            p.normals.push_back(normal);
        }
    }

    // We have generated tessU + 1 vertices per latitude.
    const unsigned int columns = tessU + 1;

    // Calculate indices.
    for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
    {
        for (unsigned int longitude = 0; longitude < tessU; ++longitude)
        {
            p.indices.push_back(latitude * columns + longitude);  // lower left
            p.indices.push_back(latitude * columns + longitude + 1);  // lower right
            p.indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 

            p.indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
            p.indices.push_back((latitude + 1) * columns + longitude);  // upper left
            p.indices.push_back(latitude * columns + longitude);  // lower left
        }
    }

    p.index = numberOfSphere++;
	p.type = "Sphere";
    p.instanceIndex = numberOfMesh++;
    p.vertex_count = p.vertices.size();
    p.transform = getInitialTransform(position);

    m_primitives.push_back(p);

    return p;
}

Primitive Primitives::createLoadMesh(std::string filename, float3 position)
{
    Primitive p;

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = -attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                tinyobj::real_t ny = -attrib.normals[3 * size_t(idx.normal_index) + 1];
                tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                float3 vs = make_float3(vx, vy, vz);
                float3 ns = make_float3(nx, ny, nz);

                p.vertices.push_back(vs);
                p.normals.push_back(ns);
                p.indices.push_back(index_offset + v);
            }
            index_offset += fv;
        }
    }

    p.index = numberOfLoaded++;
    p.type = "LoadedMesh";
    p.instanceIndex = numberOfMesh++;
    p.vertex_count = p.vertices.size();
    p.transform = getInitialTransform(position);

    m_primitives.push_back(p);

    return p;
}

glm::mat4 Primitives::getInitialTransform(float3 position)
{
    glm::mat4 t       = glm::translate(glm::mat4(1.0f), glm::vec3(position.x, position.y, position.z));
    glm::mat4 r_yaw   = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 r_pitch = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 r_roll  = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 r       = r_yaw * r_pitch * r_roll;
    glm::mat4 s       = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	glm::mat4 transform = t * r * s;

    return transform;
}