#include <gtc/matrix_transform.hpp>

#include "Plane.h"

Plane::~Plane()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
}

void Plane::setOptixBuildInput()
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

	plane_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	plane_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	plane_input.triangleArray.vertexStrideInBytes = sizeof(float3);
	plane_input.triangleArray.numVertices         = vertices.size();
	plane_input.triangleArray.vertexBuffers       = &d_vertices;

	plane_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	plane_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
	plane_input.triangleArray.numIndexTriplets   = (unsigned int)indices.size() / 3;
	plane_input.triangleArray.indexBuffer        = d_indices;

	plane_input.triangleArray.flags         = triangleInputFlags;
	plane_input.triangleArray.numSbtRecords = 1;

	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

void Plane::createTraversableHandle(OptixDeviceContext& ctx)
{
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		ctx,
		&accel_options,
		&plane_input,
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
		&plane_input,
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

void Plane::createOptixInstance(unsigned int id, float tx, float ty, float tz, float yaw, float pitch, float roll)
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
	m_instance.instanceId        = id;
	m_instance.visibilityMask    = 255;
	m_instance.sbtOffset         = 0;
	m_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
	m_instance.traversableHandle = m_gas;
}