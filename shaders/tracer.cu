#include <optix.h>

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#include "tracer.cuh"

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
	payload.t_hit            = 0.0f;
	payload.numBounces       = 0;
	payload.hitNormal        = make_float3(0.0f);
	payload.accumColor       = make_float3(0.0f);
	payload.accumAlpha       = 0.0f;
	payload.blockingRadiance = 0.0f;
	payload.directLight      = make_float3(0.0f);

	RayData rayData;
	rayData.initialize();
	payload.rayData = &rayData;
	
	setNextTraceState(TraceGaussianPass);

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

	unsigned int timeout = 0;
	while ((length(payload.currRayDirection) > 0.1f) && (payload.numBounces < MAX_BOUNCES)) {
		const float3 ray_o = payload.currRayOrigin;
		const float3 ray_d = payload.currRayDirection;

		traceMesh(ray_o, ray_d, &payload);
		
		if (getNextTraceState() == TraceTerminate) break;
		
		float4 gsRadDns;
		if (getNextTraceState() == TraceLastGaussianPass) {
			gsRadDns = traceGaussians(rayData, 
										ray_o, 
										ray_d, 
										params.t_min, 
										params.t_max, 
										&payload);
			float3 radiance = make_float3(gsRadDns.x, 
							  			  gsRadDns.y, 
										  gsRadDns.z);
			float alpha = gsRadDns.w;

			payload.directLight = radiance * alpha;
			payload.accumAlpha = clamp(payload.accumAlpha + alpha , 0.0f, 1.0f);
			setNextTraceState(TraceTerminate);
		}
		else {
			gsRadDns = traceGaussians(rayData, 
										ray_o, 
										ray_d, 
										params.t_min, 
										payload.t_hit, 
										&payload);
			float3 radiance = make_float3(gsRadDns.x, 
										  gsRadDns.y, 
										  gsRadDns.z);
			float alpha = gsRadDns.w;
			
			payload.accumColor += make_float3(1.0f - payload.accumAlpha) * radiance;
			payload.accumAlpha = clamp(payload.accumAlpha + alpha, 0.0f, 1.0f);
			payload.blockingRadiance = clamp(payload.blockingRadiance + alpha, 0.0f, 1.0f);
		}

		payload.accumColor += payload.directLight * (1.0f - payload.blockingRadiance);

		timeout += 1;
		if (timeout > TIMEOUT_ITERATIONS)
        	break;
	}

	float3 rgb = make_float3(payload.accumColor.x, payload.accumColor.y, payload.accumColor.z);
	writeOutputBuffer(rgb);
}

extern "C" __global__ void __miss__miss()
{
	if (getNextTraceState() == TraceMeshPass) {
		RayPayload* payload = getRayPayLoad();

		payload->currRayOrigin    = make_float3(0.0f);
		payload->currRayDirection = make_float3(0.0f);

		setNextTraceState(TraceLastGaussianPass);
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
	unsigned int nextState = getNextTraceState();

	float  t_hit = optixGetRayTmax();
	float3 ray_o = optixGetWorldRayOrigin();
	float3 ray_d = optixGetWorldRayDirection();

	Mesh hitMesh  = params.d_meshes[optixGetInstanceId()];
	float3 normal = getBarycentricNormal(hitMesh);

	float3 newRayDirection = make_float3(0.0f);
	nextState = TraceGaussianPass;

	if (params.type == MIRROR)
		renderMirror(ray_d, normal, newRayDirection, numBounces);
	else if (params.type == NORMAL) {
		renderNormal(ray_o, ray_d, normal, t_hit, nextState, payload);
	}
	else if (params.type == GLASS) {
		renderGlass(ray_d, normal, newRayDirection, t_hit, numBounces);
	}

	payload->t_hit            = t_hit;
	payload->currRayOrigin    = ray_o + t_hit * ray_d;
	payload->currRayDirection = newRayDirection;
	payload->hitNormal        = normal;
	payload->numBounces       = numBounces;

	setNextTraceState(nextState);
}