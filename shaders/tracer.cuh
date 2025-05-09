#pragma once

#include <optix.h>

#include "../src/Parameters.h"
#include "../src/vector_math.h"

// Reference: 3DGRUT (https://github.com/nv-tlabs/3dgrut)
constexpr float TRACE_MESH_TMIN                    = 1e-5;
constexpr float TRACE_MESH_TMAX                    = 1e5;
static    constexpr unsigned int MaxNumHitPerTrace = 7;
constexpr uint32_t TIMEOUT_ITERATIONS              = 1000;
constexpr uint32_t MAX_BOUNCES                     = 32;
constexpr float REFRACTION_EPS_SHIFT			   = 1e-5f;

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
		radiance           = make_float3(0.0f);
		density            = 0.0;
		normal             = make_float3(0.f);
		hitDistance        = 0.f;
		rayLastHitDistance = 0.f;
		hitCount           = 0.f;
	}
};

struct RayPayload
{
	float        t_hit;
	float3       currRayOrigin;
	float3       currRayDirection;
	float3       hitNormal;
	unsigned int numBounces;
	float3       accumColor;
	float        accumAlpha;
	float3  	 directLight;
	float 		 blockingRadiance;

	RayData* rayData;
};

struct HitPayload
{
	unsigned int particleId;
	float        distance;

	static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
	static constexpr float        InfiniteDistance = 1e20f;
};
using GaussianPayload = HitPayload[MaxNumHitPerTrace];

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

static __forceinline__ __device__ void setNextTraceState(unsigned int state)
{
	const uint3 idx = optixGetLaunchIndex();
	uint32_t px = idx.x;
    uint32_t py = idx.y;
    uint32_t pixelIndex = py * params.width + px;

	params.traceState[pixelIndex] = state;
}

static __forceinline__ __device__ unsigned int getNextTraceState()
{
	const uint3 idx = optixGetLaunchIndex();
	uint32_t px = idx.x;
	uint32_t py = idx.y;
	uint32_t pixelIndex = py * params.width + px;
	return params.traceState[pixelIndex];
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

static __forceinline__ __device__ float computeResponse(GaussianParticle& gp, 
                                                        const float3& o, 
                                                        const float3& d)
{
    glm::vec3 mu = glm::vec3(gp.position.x, gp.position.y, gp.position.z);

    glm::mat3 R = glm::mat3_cast(gp.rotation);
    glm::mat3 RT = glm::transpose(R);

    glm::mat3 inv_s(1.0f);
    inv_s[0][0] = 1.0f / gp.scale.x;
    inv_s[1][1] = 1.0f / gp.scale.y;
    inv_s[2][2] = 1.0f / gp.scale.z;

    glm::mat3 invCov = inv_s * RT;

    glm::vec3 _o = glm::vec3(o.x, o.y, o.z);
    glm::vec3 _d = glm::vec3(d.x, d.y, d.z);

    glm::vec3 o_g = invCov * (_o - mu);
    glm::vec3 d_g = invCov * _d;

    float d_val = -glm::dot(o_g, d_g) / fmaxf(1e-6f, glm::dot(d_g, d_g));
    glm::vec3 pos = _o + d_val * _d;
    glm::vec3 p_g = invCov * (mu - pos);

    return exp(-0.5f * glm::dot(p_g, p_g));
}

static __forceinline__ __device__ float3 SHToRadiance(GaussianParticle& gp, float3& d)
{
    float3 sh[16];
    for (int i = 0; i < 16; i++) {
        sh[i] = gp.sh[i];
    }

    float3 L = make_float3(0.5f) + SH_C0 * sh[0];
    if (params.sh_degree_max == 0)
        return L;

    float x = d.x;
    float y = d.y;
    float z = d.z;
    L += SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);
    if (params.sh_degree_max == 1)
        return L;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    L +=
        SH_C2_0 * xy * sh[4] +
        SH_C2_1 * yz * sh[5] +
        SH_C2_2 * (2. * zz - xx - yy) * sh[6] +
        SH_C2_3 * xz * sh[7] +
        SH_C2_4 * (xx - yy) * sh[8];
    if (params.sh_degree_max == 2)
        return L;

    L +=
        SH_C3_0 * y * (3.0f * xx - yy) * sh[9] +
        SH_C3_1 * xy * z * sh[10] +
        SH_C3_2 * y * (4.0f * zz - xx - yy) * sh[11] +
        SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
        SH_C3_4 * x * (4.0f * zz - xx - yy) * sh[13] +
        SH_C3_5 * z * (xx - yy) * sh[14] +
        SH_C3_6 * x * (xx - 3.0f * yy) * sh[15];
    return L;
}

static __forceinline__ __device__ float3 computeRadiance(GaussianParticle& gp, float3& d)
{
    float3 L = SHToRadiance(gp, d);
    return fmaxf(L, make_float3(0.0f));
}

static __forceinline__ __device__ void traceMesh(float3 ray_origin, float3 ray_direction, RayPayload* payload)
{
	setNextTraceState(TraceMeshPass);

	uint32_t u0, u1;
	packPointer(payload, u0, u1);

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
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
    r0 = r2 = r4 = r6 = r8 = r10 = r12 = HitPayload::InvalidParticleId;
    r1 = r3 = r5 = r7 = r9 = r11 = r13 = __float_as_int(HitPayload::InfiniteDistance);

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
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13);

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
}

static __forceinline__ __device__ void trace(RayData& rayData,
	                                         const float3& ray_o,
	                                         const float3& ray_d,
	                                         const float t_min,
	                                         const float t_max)
{
	float rayTransmittance   = 1.0f - rayData.density;
	constexpr float epsT     = 1e-9;
	float rayLastHitDistance = t_min;
    
    float3 radiance = make_float3(0.0f);

	GaussianPayload gaussianPayload;
	while ((rayLastHitDistance <= t_max) && (rayTransmittance > params.minTransmittance)) {
		traceGPs(gaussianPayload, ray_o, ray_d, rayLastHitDistance + epsT, t_max + epsT);

		if (gaussianPayload[0].particleId == HitPayload::InvalidParticleId) {
			break;
		}

#pragma unroll
		for (int i = 0; i < MaxNumHitPerTrace; i++) {
			const HitPayload hit = gaussianPayload[i];
            GaussianParticle currParticle = params.d_particles[hit.particleId];

            if ((hit.particleId != HitPayload::InvalidParticleId) && (rayTransmittance > params.minTransmittance)) {
                rayLastHitDistance = fmaxf(hit.distance, rayLastHitDistance);

                float hitAlpha = computeResponse(currParticle, ray_o, ray_d);
                hitAlpha = fminf(0.99f, hitAlpha * currParticle.opacity);

                float3 hitRadiance = computeRadiance(currParticle, normalize(ray_d));

                if (params.alpha_min < hitAlpha) {
                    float3 hitRadiance = computeRadiance(currParticle, normalize(ray_d));

                    radiance += rayTransmittance * hitRadiance * hitAlpha;
                    rayTransmittance *= (1.0f - hitAlpha);
                }
            }
		}
	}

    rayData.radiance = radiance;
    rayData.density  = 1.0f - rayTransmittance;
}

static __forceinline__ __device__ float4 traceGaussians(RayData& rayData,
	                                                    const float3& ray_o,
	                                                    const float3& ray_d,
	                                                    const float t_min,
	                                                    const float t_max,
	                                                    RayPayload* payload)
{
	setNextTraceState(TraceGaussianPass);

	trace(rayData, ray_o, ray_d, t_min, t_max);

    float4 accumulated_radiance = make_float4(
        rayData.radiance.x,
        rayData.radiance.y,
        rayData.radiance.z,
		rayData.density
    );

    return accumulated_radiance;
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

static __forceinline__ __device__ void renderNormal(const float3 ray_o,
													const float3 ray_d,
													const float3 normal,
													float& t_hit,
													unsigned int& traceState,
													RayPayload* payload)
{
	const float4 gsRadDns = traceGaussians(payload->rayData[0], 
										   ray_o, 
										   ray_d, 
										   params.t_min, 
										   t_hit, 
										   payload);
	float3 radiance = make_float3(gsRadDns.x, gsRadDns.y, gsRadDns.z);
	float alpha = gsRadDns.w;
	payload->accumColor += radiance;
	payload->accumAlpha += alpha;

	const float3 normalColor = (normal + 1) / 2;
	payload->accumColor += normalColor * (1.0f - alpha);
	payload->accumAlpha += (1.0f - alpha);

	traceState = TraceTerminate;
}

// TODO: Add safe_normalize
static __forceinline__ __device__ void refract(float3& newRayDirction,
											   const float3 ray_d,
											   float3 normal,
											   const float etai_over_etat,
											   float& t_hit,
											   unsigned int& numBounces)
{
	float ri;
	if (dot(ray_d, normal) < 0.0f) {
		ri = 1.0f / etai_over_etat;
	}
	else {
		ri = etai_over_etat;
		normal = -normal;
	}

	float cos_theta = fminf(dot(-ray_d, normal), 1.0f);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	bool cannot_refract = ri * sin_theta > 1.0f;
	if (cannot_refract) {
		// Reflect
		float3 reflected_normal = dot(ray_d, normal) < 0.0f ? normal : -normal;
		newRayDirction = reflect(ray_d, reflected_normal);
		numBounces += 1;
	}
	else {
		// Refract
		float3 r_out_perp = ri * (ray_d + cos_theta * normal);
		float3 r_out_parallel = -sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp))) * normal;
		newRayDirction = r_out_perp + r_out_parallel;
		t_hit += REFRACTION_EPS_SHIFT;
	}
}

static __forceinline__ __device__ void renderGlass(const float3 ray_d,
												   float3 normal,
												   float3& newRayDirction,
												   float& t_hit,
												   unsigned int& numBounces)
{
	const Mesh hitMesh  = params.d_meshes[optixGetInstanceId()];
	const unsigned int primitive_index = optixGetPrimitiveIndex();
	
	// TODO: Set n2 in host code.
	float n1 = 1.0003f; // Air
	float n2 = 1.5f;    // Glass
	float etai_over_etat = n2 / n1;

	refract(newRayDirction, ray_d, normal, etai_over_etat, t_hit, numBounces);
}

static __forceinline__ __device__ void writeOutputBuffer(float3 rgb)
{
	const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

	rgb = clamp(rgb, 0.0f, 1.0f);

    params.output_buffer[image_index] = make_uchar3(
        quantizeUnsigned8Bits(rgb.x),
        quantizeUnsigned8Bits(rgb.y),
        quantizeUnsigned8Bits(rgb.z)
    );
}