#pragma once

#include <optix.h>

#include "../src/Parameters.h"
#include "../src/vector_math.h"

extern "C"
{
	__constant__ Params params;
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& u0, uint32_t& u1)
{
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	u0 = uptr >> 32;
	u1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__ void getRay(const uint3 idx,
											  const float3 U, 
											  const float3 V, 
											  const float3 W, 
											  const float3 eye,	
											  const int width,
											  const int height,		
											  float3& ray_origin,
											  float3& ray_direction)
{
	// TODO: Check jitter
	const float2 subpixel_jitter = make_float2(0.5f, 0.5f);
	const float2 d = 2.0f * make_float2(
		(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(width),
		(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(height)
	) - 1.0f;

	ray_origin    = eye;
	ray_direction = normalize(d.x * U + d.y * V + W);
}

static __forceinline__ __device__ void getFishEyeRay(const uint3 idx,
			  										 const float3 U,
													 const float3 V,
													 const float3 W,
													 const float3 eye,
													 const int width,
													 const int height,
													 float3& ray_origin,
													 float3& ray_direction)
{
	// TODO: Check jitter
	const float2 subpixel_jitter = make_float2(0.5f, 0.5f);
	const float2 d = 2.0f * make_float2(
		(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(width),
		(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(height)
	) - 1.0f;

	float r = sqrtf(d.x * d.x + d.y * d.y);

	if (r > 1.0f) return;

	const float maxTheta = M_PI / 2.0f;
	float f     = 1.0f / sqrtf(2.0f);
	float theta = 2.0f * asinf(r / (2.0f * f));
	float phi   = atan2f(d.y, d.x);
	float3 direction = make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));

	ray_origin    = eye;
	ray_direction = normalize(direction.x * U + direction.y * V + direction.z * W);
}

static __forceinline__ __device__ float3 getBarycentricNormal(Mesh mesh)
{
	unsigned int primitive_index = optixGetPrimitiveIndex();

	uint3 face = mesh.faces[primitive_index];

	float3 n0 = mesh.vertex_normals[face.x];
	float3 n1 = mesh.vertex_normals[face.y];
	float3 n2 = mesh.vertex_normals[face.z];

	float2 barycentrics = optixGetTriangleBarycentrics();

	float w0 = 1.0f - barycentrics.x - barycentrics.y;
	float w1 = barycentrics.x;
	float w2 = barycentrics.y;

	float3 normal = normalize(w0 * n0 + w1 * n1 + w2 * n2);
	return normal;
}