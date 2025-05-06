#include <optix.h>

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#include "tracer.cuh"

constexpr uint32_t MAX_BOUNCES = 32;

extern "C"
{
	#ifndef __PARAMS__
		__constant__ Params params;
	#define __PARAMS__ 1
	#endif
}

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
	while ((length(payload.currRayDirection) > 0.1f) && (payload.numBounces < MAX_BOUNCES)) {
		const float3 ray_o = payload.currRayOrigin;
		const float3 ray_d = payload.currRayDirection;
		
		traceMesh(ray_o, ray_d, &payload);
		

	}

//     float3 result;
//     unsigned int timeout = 0;
// 	//while ((length(payload.currRayDirection) > 0.1f) && (payload.numBounces < MAX_BOUNCES)) {
// 	const float3 ray_o = payload.currRayOrigin;
// 	const float3 ray_d = payload.currRayDirection;
		
// 	traceMesh(ray_o, ray_d, &payload);

//     if (payload.t_hit == 0.0f) payload.t_hit = params.t_max;

// 	//if (getNextTraceState() == TraceTerminate) break;
//     //if (payload.traceState == TraceTerminate) break;

//     traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
//     result = make_float3(payload.rayData->radiance.x, payload.rayData->radiance.y, payload.rayData->radiance.z);
//         //float4 gaussianRadDns;
//         //if (payload.traceState == TraceLastGaussianPass) {
//         //    result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);
//         //    payload.traceState = TraceTerminate;
//         //}
//         //else {
//         //    result = traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
//         //}
// 		//float4 gaussianRadiance;
// 		//if (getNextTraceState() == TraceLastGaussianPass) {
//   //      if (payload.traceState == TraceLastGaussianPass) {
// 		//	result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);
// 		//}
// 		//else {
// 		//	result = traceGaussians(rayData, ray_o, ray_d, 1e-9, payload.t_hit, &payload);
// 		//}

//         //result = traceGaussians(rayData, ray_o, ray_d, 1e-9, params.t_max, &payload);

//  //       timeout += 1;
//  //       if (timeout > TIMEOUT_ITERATIONS)
//  //           break;
// 	//}

//     // const uint3    launch_index = optixGetLaunchIndex();
//     // const unsigned int image_index = launch_index.y * params.width + launch_index.x;
//     // float3 accum_color = make_float3(result.x, result.y, result.z);

//     // float3 rgb = clamp(accum_color, 0.0f, 1.0f);

//     // params.output_buffer[image_index] = make_uchar3(
//     //     quantizeUnsigned8Bits(rgb.x),
//     //     quantizeUnsigned8Bits(rgb.y),
//     //     quantizeUnsigned8Bits(rgb.z)
//     // );

// 	writeOutputBuffer(result);
	
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
    RayPayload* payload = getRayPayLoad();

    if (payload->traceState == TraceMeshPass) {
       payload->currRayOrigin    = make_float3(0.0f);
       payload->currRayDirection = make_float3(0.0f);
       payload->traceState       = TraceLastGaussianPass;
    }
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