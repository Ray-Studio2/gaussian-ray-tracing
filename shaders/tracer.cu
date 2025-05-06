#include <optix.h>

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#include "tracer.cuh"

constexpr uint32_t MAX_BOUNCES = 32;

//static __forceinline__ __device__ void swap(HitInfo& a, HitInfo& b)
//{
//	HitInfo temp = a;
//	a = b;
//	b = temp;
//}

extern "C"
{
	#ifndef __PARAMS__
		__constant__ Params params;
	#define __PARAMS__ 1
	#endif
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

//static __forceinline__ __device__ float3 SHToRadiance(GaussianParticle& gp, float3& d)
//{
//	float3 sh[16];
//	for (int i = 0; i < 16; i++) {
//		sh[i] = gp.sh[i];
//	}
//
//	float3 L = make_float3(0.5f) + SH_C0 * sh[0];
//	if (params.sh_degree_max == 0)
//		return L;
//
//	float x = d.x;
//	float y = d.y;
//	float z = d.z;
//	L += SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);
//	if (params.sh_degree_max == 1)
//		return L;
//
//	float xx = x * x;
//	float yy = y * y;
//	float zz = z * z;
//	float xy = x * y;
//	float xz = x * z;
//	float yz = y * z;
//	L +=
//		SH_C2_0 * xy * sh[4] +
//		SH_C2_1 * yz * sh[5] +
//		SH_C2_2 * (2. * zz - xx - yy) * sh[6] +
//		SH_C2_3 * xz * sh[7] +
//		SH_C2_4 * (xx - yy) * sh[8];
//	if (params.sh_degree_max == 2)
//		return L;
//
//	L +=
//		SH_C3_0 * y * (3.0f * xx - yy) * sh[9] +
//		SH_C3_1 * xy * z * sh[10] +
//		SH_C3_2 * y * (4.0f * zz - xx - yy) * sh[11] +
//		SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
//		SH_C3_4 * x * (4.0f * zz - xx - yy) * sh[13] +
//		SH_C3_5 * z * (xx - yy) * sh[14] +
//		SH_C3_6 * x * (xx - 3.0f * yy) * sh[15];
//	return L;
//}

//static __forceinline__ __device__ float computeResponse(
//	GaussianParticle& gp,
//	const float3& o,
//	const float3& d
//)
//{
//	glm::vec3 mu = glm::vec3(gp.position.x, gp.position.y, gp.position.z);
//
//	glm::mat3 R = glm::mat3_cast(gp.rotation);
//	glm::mat3 RT = glm::transpose(R);
//
//	glm::mat3 inv_s(1.0f);
//	inv_s[0][0] = 1.0f / gp.scale.x;
//	inv_s[1][1] = 1.0f / gp.scale.y;
//	inv_s[2][2] = 1.0f / gp.scale.z;
//
//	glm::mat3 invCov = inv_s * RT;
//
//	glm::vec3 _o = glm::vec3(o.x, o.y, o.z);
//	glm::vec3 _d = glm::vec3(d.x, d.y, d.z);
//
//	glm::vec3 o_g = invCov * (_o - mu);
//	glm::vec3 d_g = invCov * _d;
//
//	float d_val = -glm::dot(o_g, d_g) / fmaxf(1e-6f, glm::dot(d_g, d_g));
//	glm::vec3 pos = _o + d_val * _d;
//	glm::vec3 p_g = invCov * (mu - pos);
//
//	return exp(-0.5f * glm::dot(p_g, p_g));
//}

//static __forceinline__ __device__ float3 computeRadiance(GaussianParticle& gp, float3& d)
//{
//	float3 L = SHToRadiance(gp, d);
//	return make_float3(fmaxf(L.x, 0.0f), fmaxf(L.y, 0.0f), fmaxf(L.z, 0.0f));
//}

//static __forceinline__ __device__ float3 trace(
//	OptixTraversableHandle handle,
//	const float3 ray_origin,
//	const float3 ray_direction,
//	RayPayload* prd
//)
//{
//    float3 L = make_float3(0.0f);
//    float  T = 1;
//    float  t_curr = params.t_min;
//    const  float epsilon = 1e-4f;
//
//	prd->hit_count = 0;
//	prd->hit_reflection_primitive = false;
//
//	traceMesh(ray_origin, ray_direction, prd);
//
//	while (params.minTransmittance < T && t_curr < params.t_max) {
//		for (int i = 0; i < params.k; i++) {
//			prd->k_closest[i].t = params.t_max;
//			prd->k_closest[i].particleIndex = -1;
//		}
//
//
//		uint32_t u0, u1;
//		packPointer(prd, u0, u1);
//
//		optixTrace(
//			handle,
//			ray_origin,
//			ray_direction,
//			t_curr,
//			params.t_max,
//			0.0f,
//			OptixVisibilityMask(1),
//			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
//			0,        // SBT offset
//			1,        // SBT stride
//			0,        // missSBTIndex
//			u0, u1
//		);
//
//		t_curr = prd->k_closest[params.k - 1].t + epsilon;
//
//#pragma unroll
//		for (int i = 0; i < params.k; i++)
//		{
//			if (prd->k_closest[i].particleIndex == -1)
//			{
//				t_curr = params.t_max;
//				break;
//			}
//
//			GaussianParticle gp = params.d_particles[prd->k_closest[i].particleIndex];
//
//			float alpha_hit = computeResponse(gp, ray_origin, ray_direction);
//			alpha_hit = fminf(0.99f, alpha_hit * gp.opacity);
//
//			if (params.alpha_min < alpha_hit)
//			{
//				float3 pos = make_float3(gp.position.x, gp.position.y, gp.position.z);
//				float3 L_hit = computeRadiance(gp, normalize(ray_direction));
//
//				L += T * alpha_hit * L_hit;
//				T *= 1.0f - alpha_hit;
//			}
//		}
//	}
//
//	if (prd->hit_reflection_primitive) {
//		if (prd->t_hit_reflection < t_curr)
//			return make_float3(0.0f, 0.0f, 0.0f);
//		else
//			prd->hit_reflection_primitive = false;
//	}
//
//	return make_float3(L.x, L.y, L.z);
//}

extern "C" __global__ void __raygen__raygeneration()
{
	// Initialize payload.
	RayPayload payload;
	payload.t_hit      = 0.0f;
	payload.numBounces = 0;
	payload.hitNormal  = make_float3(0.0f);

	RayData rayData;
	rayData.initialize();
	payload.rayData = &rayData;

	//setNextTraceState(TraceGaussianPass);
    payload.traceState = TraceGaussianPass;

	if (!params.mode_fisheye) {
		getRay(optixGetLaunchIndex(),
			   -params.U,
			   -params.V,
			   params.W,
			   params.eye,
			   params.width,
			   params.height,
			   payload.currRayOrigin,
			   payload.currRayDirection);
	}
	else {
		getFishEyeRay(optixGetLaunchIndex(),
					  -params.U,
					  -params.V,
					  params.W,
					  params.eye,
					  params.width,
					  params.height,
					  payload.currRayOrigin,
					  payload.currRayDirection);
	}

	// TODO: 3DGRUT trace state.
    float3 result;
    unsigned int timeout = 0;
	//while ((length(payload.currRayDirection) > 0.1f) && (payload.numBounces < MAX_BOUNCES)) {
	const float3 ray_o = payload.currRayOrigin;
	const float3 ray_d = payload.currRayDirection;
		
	traceMesh(ray_o, ray_d, &payload);

    if (payload.t_hit == 0.0f) payload.t_hit = params.t_max;

	//if (getNextTraceState() == TraceTerminate) break;
    //if (payload.traceState == TraceTerminate) break;

    traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
    result = make_float3(payload.rayData->radiance.x, payload.rayData->radiance.y, payload.rayData->radiance.z);
        //float4 gaussianRadDns;
        //if (payload.traceState == TraceLastGaussianPass) {
        //    result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);
        //    payload.traceState = TraceTerminate;
        //}
        //else {
        //    result = traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
        //}
		//float4 gaussianRadiance;
		//if (getNextTraceState() == TraceLastGaussianPass) {
  //      if (payload.traceState == TraceLastGaussianPass) {
		//	result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);
		//}
		//else {
		//	result = traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
		//}

        //result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);

 //       timeout += 1;
 //       if (timeout > TIMEOUT_ITERATIONS)
 //           break;
	//}

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3 accum_color = make_float3(result.x, result.y, result.z);

    float3 rgb = clamp(accum_color, 0.0f, 1.0f);

    params.output_buffer[image_index] = make_uchar3(
        quantizeUnsigned8Bits(rgb.x),
        quantizeUnsigned8Bits(rgb.y),
        quantizeUnsigned8Bits(rgb.z)
    );


	//float3 result = make_float3(0.0f);

	//RayPayload prd;
	//int recursion_count = 0;
	//result = trace(params.handle, ray_origin, ray_direction, &prd);

	//while (recursion_count < MAX_BOUNCES) {
	//	result = trace(params.handle, ray_origin, ray_direction, &prd);
	//	if (!prd.hit_reflection_primitive) {
	//		break;
	//	}

	//	if (params.reflection_render_normals) {
	//		result = (prd.hit_normal + 1) / 2;
	//		break;
	//	}

	//	if (length(prd.hit_position - ray_origin) < 1e-6){
	//		break;
	//	}

	//	ray_origin = prd.hit_position;
	//	ray_direction = reflect(ray_direction, prd.hit_normal);
	//	recursion_count++;
	//}
	//
	//const uint3    launch_index = optixGetLaunchIndex();
	//const unsigned int image_index = launch_index.y * params.width + launch_index.x;
	//float3 accum_color = result;

	//float3 rgb = clamp(accum_color, 0.0f, 1.0f);

	//params.output_buffer[image_index] = make_uchar3(
	//	quantizeUnsigned8Bits(rgb.x),
	//	quantizeUnsigned8Bits(rgb.y),
	//	quantizeUnsigned8Bits(rgb.z)
	//);
}

extern "C" __global__ void __miss__miss()
{
    //RayPayload* payload = getRayPayLoad();

    //if (payload->traceState == TraceMeshPass) {
    //    payload->currRayOrigin    = make_float3(0.0f);
    //    payload->currRayDirection = make_float3(0.0f);
    //    payload->traceState       = TraceLastGaussianPass;
    //}
}

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                      \
    {                                                                             \
        const float distance = __uint_as_float(optixGetPayload_##i_distance##()); \
        if (hit.distance < distance) {                                            \
            optixSetPayload_##i_distance##(__float_as_uint(hit.distance));        \
            const uint32_t id = optixGetPayload_##i_id##();                       \
            optixSetPayload_##i_id##(hit.particleId);                             \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

extern "C" __global__ void __anyhit__anyhit()
{
	HitPayload hit = HitPayload{ optixGetInstanceId(), optixGetRayTmax() };
	if (hit.distance < __uint_as_float(optixGetPayload_13())) {
		compareAndSwapHitPayloadValue(hit, 0, 1);
		compareAndSwapHitPayloadValue(hit, 2, 3);
		compareAndSwapHitPayloadValue(hit, 4, 5);
		compareAndSwapHitPayloadValue(hit, 6, 7);
		compareAndSwapHitPayloadValue(hit, 8, 9);
		compareAndSwapHitPayloadValue(hit, 10, 11);
		compareAndSwapHitPayloadValue(hit, 12, 13);

		// ignore all inserted hits, expect if the last one
		if (__uint_as_float(optixGetPayload_13()) > optixGetRayTmax()) {
			optixIgnoreIntersection();
		}
	}
}

extern "C" __global__ void __closesthit__closesthit()
{
	RayPayload* payload = getRayPayLoad();
	unsigned int numBounces = payload->numBounces;
	//unsigned int nextTraceState = getNextTraceState();
    unsigned int nextTraceState = payload->traceState;

	float  t_hit = optixGetRayTmax();
	float3 ray_o = optixGetWorldRayOrigin();
	float3 ray_d = optixGetWorldRayDirection();

	Mesh hitMesh  = params.d_meshes[optixGetInstanceId()];
	float3 normal = getBarycentricNormal(hitMesh);

	float3 newRayDirection = make_float3(0.0f);
	nextTraceState = TraceGaussianPass;

	if (params.type == MIRROR)
		renderMirror(ray_d, normal, newRayDirection, numBounces);

	payload->t_hit            = t_hit;
	payload->currRayOrigin    = ray_o + t_hit * ray_d;
	payload->currRayDirection = newRayDirection;
	payload->hitNormal        = normal;
	payload->numBounces       = numBounces;

	//setNextTraceState(nextTraceState);
    payload->traceState = nextTraceState;
	 
	//payload->hit_count++;
	//payload->hit_reflection_primitive = true;
	//payload->t_hit_reflection = t_hit;

	//Mesh hitMesh = params.d_meshes[optixGetInstanceId()];
	//float3 hit_normal = getBarycentricNormal(hitMesh);

	//payload->hit_normal = hit_normal;
	//payload->hit_position = ray_d * t_hit + ray_o;
}