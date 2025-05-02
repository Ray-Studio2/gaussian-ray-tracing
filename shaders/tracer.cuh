#pragma once

#include <optix.h>

#include "../src/Parameters.h"
#include "../src/vector_math.h"

// Reference: 3DGRUT (https://github.com/nv-tlabs/3dgrut)
constexpr float TRACE_MESH_TMIN = 1e-5;
constexpr float TRACE_MESH_TMAX = 1e5;

extern "C"
{
	#ifndef __PARAMS__
		__constant__ Params params;
	#define __PARAMS__ 1
	#endif
}

struct RayData
{
	float3 radiance;
	float  density;
	float3 normal;
	float  hitDistance;
	float  rayLastHitDistance;
	float  hitCount;

	__device__ void initialize() {
		radiance = make_float3(0.0f);
		density = 0.0;
		normal = make_float3(0.f);
		hitDistance = 0.f;
		rayLastHitDistance = 0.f;
		hitCount = 0.f;
	}
};

struct RayPayload
{
	float        t_hit;
	float3       currRayOrigin;
	float3       currRayDirection;
	float3       hitNormal;
	unsigned int numBounces;

	RayData* rayData;
};

struct HitPayload
{
	unsigned int particleId;
	float        distance;

	static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
	static constexpr float        InfiniteDistance = 1e20f;
};
using GaussianPayload = HitPayload[16];

//extern "C"
//{
//	__constant__ Params params;
//}

static __forceinline__ __device__ void setNextTraceState(unsigned int traceState)
{
	params.traceState = traceState;
}

static __forceinline__ __device__ unsigned int getNextTraceState()
{
	return params.traceState;
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

static __forceinline__ __device__ RayPayload* getRayPayLoad()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<RayPayload*>(unpackPointer(u0, u1));
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

static __forceinline__ __device__ void traceMesh(float3 ray_origin, float3 ray_direction, RayPayload* prd)
{
	setNextTraceState(TraceMeshPass);
	
	uint32_t u0, u1;
	packPointer(prd, u0, u1);

	optixTrace(
		params.mesh_handle,
		ray_origin,
		ray_direction,
		TRACE_MESH_TMIN,
		TRACE_MESH_TMAX,
		0.0f,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,        // SBT offset
		1,        // SBT stride
		0,        // missSBTIndex
		u0, u1
	);
}

static __forceinline__ __device__ void traceGPs(GaussianPayload& gaussianPayload,
												const float3& ray_o,
												const float3& ray_d,
												const float t_min,
												const float t_max)
{
	uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
		r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
	r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = HitPayload::InvalidParticleId;
	r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 = __float_as_int(HitPayload::InfiniteDistance);

	optixTrace(params.handle, 
			   ray_o, 
			   ray_d,
			   t_min,
			   t_max,
			   0.0f,
			   OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			   0, // SBT offset
			   1, // SBT stride
			   0, // missSBTIndex
			   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
			   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);

	gaussianPayload[0].particleId = r0;
	gaussianPayload[0].distance = __uint_as_float(r1);
	gaussianPayload[1].particleId = r2;
	gaussianPayload[1].distance = __uint_as_float(r3);
	gaussianPayload[2].particleId = r4;
	gaussianPayload[2].distance = __uint_as_float(r5);
	gaussianPayload[3].particleId = r6;
	gaussianPayload[3].distance = __uint_as_float(r7);
	gaussianPayload[4].particleId = r8;
	gaussianPayload[4].distance = __uint_as_float(r9);
	gaussianPayload[5].particleId = r10;
	gaussianPayload[5].distance = __uint_as_float(r11);
	gaussianPayload[6].particleId = r12;
	gaussianPayload[6].distance = __uint_as_float(r13);
	gaussianPayload[7].particleId = r14;
	gaussianPayload[7].distance = __uint_as_float(r15);
	gaussianPayload[8].particleId = r16;
	gaussianPayload[8].distance = __uint_as_float(r17);
	gaussianPayload[9].particleId = r18;
	gaussianPayload[9].distance = __uint_as_float(r19);
	gaussianPayload[10].particleId = r20;
	gaussianPayload[10].distance = __uint_as_float(r21);
	gaussianPayload[11].particleId = r22;
	gaussianPayload[11].distance = __uint_as_float(r23);
	gaussianPayload[12].particleId = r24;
	gaussianPayload[12].distance = __uint_as_float(r25);
	gaussianPayload[13].particleId = r26;
	gaussianPayload[13].distance = __uint_as_float(r27);
	gaussianPayload[14].particleId = r28;
	gaussianPayload[14].distance = __uint_as_float(r29);
	gaussianPayload[15].particleId = r30;
	gaussianPayload[15].distance = __uint_as_float(r31);
}

static __forceinline__ __device__ void trace(RayData& rayData,
	const float3& ray_o,
	const float3& ray_d,
	const float t_min,
	const float t_max)
{
	float rayTransmittance = 1.0f - rayData.density;
	constexpr float epsT = 1e-9;
	float rayLastHitDistance = t_min;

	GaussianPayload gaussianPayload;
	while ((rayLastHitDistance <= t_min) && (rayTransmittance > params.minTransmittance)) {
		traceGPs(gaussianPayload, ray_o, ray_d, t_min, t_max);

		if (gaussianPayload[0].particleId == HitPayload::InvalidParticleId) {
			break;
		}

#pragma unroll
		for (int i = 0; i < 16; i++) {
			const HitPayload hit = gaussianPayload[i];
		}
	}
}

static __forceinline__ __device__ float4 traceGaussians(RayData& rayData,
	const float3& ray_o,
	const float3& ray_d,
	const float t_min,
	const float t_max,
	RayPayload* payload)
{
	RayData prevRayData = rayData;

	setNextTraceState(TraceGaussianPass);

	trace(rayData, ray_o, ray_d, t_min, t_max);
}

static __forceinline__ __device__ void renderMirror(const float3 ray_d,
													float3 normal,
													float3& newRayDirction,	
													unsigned int& numBounces)
{
	// TODO: safe normalize, check 3DGRUT reflected normal.
	newRayDirction = reflect(ray_d, normal);
	numBounces += 1;
}