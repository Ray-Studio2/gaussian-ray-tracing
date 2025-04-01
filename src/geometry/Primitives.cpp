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

	p.index = m_primitive_count++;
	p.vertex_count = p.vertices.size();
	p.transform = getInitialTransform(position);

	m_primitives.push_back(p);
	return p;
}

//void Primitives::createSphere()
//{
//    Primitive p;
//
//    // TODO: Hardcoded..
//    unsigned int tessU = 180;
//    unsigned int tessV = 90;
//    float        radius = 0.3f;
//    float        maxTheta = M_PIf;
//
//    p.vertices.reserve((tessU + 1) * tessV);
//    p.indices.reserve(6 * tessU * (tessV - 1));
//
//    float phi_step = 2.0f * M_PIf / (float)tessU;
//    float theta_step = maxTheta / (float)(tessV - 1);
//
//    // Latitudinal rings.
//    // Starting at the south pole going upwards on the y-axis.
//    for (unsigned int latitude = 0; latitude < tessV; ++latitude) // theta angle
//    {
//        float theta = (float)latitude * theta_step;
//        float sinTheta = sinf(theta);
//        float cosTheta = cosf(theta);
//
//        float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]
//
//        // Generate vertices along the latitudinal rings.
//        // On each latitude there are tessU + 1 vertices.
//        // The last one and the first one are on identical positions, but have different texture coordinates!
//        // Note that each second triangle connected to the two poles has zero area if the sphere is closed.
//        // But since this is also used for open spheres (maxTheta < 1.0f) this is required.
//        for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
//        {
//            float phi = (float)longitude * phi_step;
//            float sinPhi = sinf(phi);
//            float cosPhi = cosf(phi);
//
//            float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]
//
//            // Unit sphere coordinates are the normals.
//            float3 normal = make_float3(cosPhi * sinTheta,
//                cosTheta,           // -y to start at the south pole.
//                sinPhi * sinTheta);
//            float3 vertex;
//            vertex = normal * radius;
//            p.vertices.push_back(vertex);
//            p.normals.push_back(normal);
//        }
//    }
//
//    // We have generated tessU + 1 vertices per latitude.
//    const unsigned int columns = tessU + 1;
//
//    // Calculate indices.
//    for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
//    {
//        for (unsigned int longitude = 0; longitude < tessU; ++longitude)
//        {
//            p.indices.push_back(latitude * columns + longitude);  // lower left
//            p.indices.push_back(latitude * columns + longitude + 1);  // lower right
//            p.indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
//
//            p.indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
//            p.indices.push_back((latitude + 1) * columns + longitude);  // upper left
//            p.indices.push_back(latitude * columns + longitude);  // lower left
//        }
//    }
//
//    p.transform = getInitialTransform();
//
//    m_primitives.push_back(p);
//}

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