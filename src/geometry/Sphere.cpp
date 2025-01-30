#include <math.h>
#include <gtc/matrix_transform.hpp>

#include "Sphere.h"
#include "../vector_math.h"

Sphere::Sphere()
{
	createSphere();
}

Sphere::~Sphere()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
}

void Sphere::createSphere()
{
	// TODO: Hardcoded..
	unsigned int tessU    = 180;
	unsigned int tessV    = 90;
	float        radius   = 0.3f;
	float        maxTheta = M_PIf;

    vertices.reserve((tessU + 1) * tessV);
    indices.reserve(6 * tessU * (tessV - 1));

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
                                        -cosTheta,           // -y to start at the south pole.
                                        -sinPhi * sinTheta);
            float3 vertex;
			vertex = normal * radius;
            vertices.push_back(vertex);
        }
    }

    // We have generated tessU + 1 vertices per latitude.
    const unsigned int columns = tessU + 1;

    // Calculate indices.
    for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
    {
        for (unsigned int longitude = 0; longitude < tessU; ++longitude)
        {
            indices.push_back(latitude * columns + longitude);  // lower left
            indices.push_back(latitude * columns + longitude + 1);  // lower right
            indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 

            indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
            indices.push_back((latitude + 1) * columns + longitude);  // upper left
            indices.push_back(latitude * columns + longitude);  // lower left
        }
    }
}

void Sphere::setOptixBuildInput()
{
    const size_t vertices_size_in_bytes = vertices.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    const size_t indices_size_in_bytes = indices.size() * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(),
        indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    sphere_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    sphere_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    sphere_input.triangleArray.numVertices         = vertices.size();
    sphere_input.triangleArray.vertexBuffers       = &d_vertices;

    sphere_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    sphere_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    sphere_input.triangleArray.numIndexTriplets   = (unsigned int)indices.size() / 3;
    sphere_input.triangleArray.indexBuffer        = d_indices;

    sphere_input.triangleArray.flags = triangleInputFlags;
    sphere_input.triangleArray.numSbtRecords = 1;

    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

void Sphere::createTraversableHandle(OptixDeviceContext& ctx)
{
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        ctx,
        &accel_options,
        &sphere_input,
        1,
        &gas_buffer_sizes
    ));

    CUDA_CHECK(cudaMalloc((void**)&d_gas, gas_buffer_sizes.outputSizeInBytes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        ctx,
        0,
        &accel_options,
        &sphere_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas,
        gas_buffer_sizes.outputSizeInBytes,
        &m_gas,
        0,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer));
}

void Sphere::createOptixInstance(unsigned int id, float tx, float ty, float tz, float yaw, float pitch, float roll)
{
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, tz));

    glm::mat4 Ryaw = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 Rpitch = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 Rroll = glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 rotation = Rroll * Rpitch * Ryaw;

    glm::mat4 transform = translation * rotation;

    float instance_transform[12] = {
        transform[0][0], transform[1][0], transform[2][0], transform[3][0],
        transform[0][1], transform[1][1], transform[2][1], transform[3][1],
        transform[0][2], transform[1][2], transform[2][2], transform[3][2]
    };

    memcpy(m_instance.transform, instance_transform, sizeof(float) * 12);
    m_instance.instanceId = id;
    m_instance.visibilityMask = 255;
    m_instance.sbtOffset = 0;
    m_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    m_instance.traversableHandle = m_gas;
}