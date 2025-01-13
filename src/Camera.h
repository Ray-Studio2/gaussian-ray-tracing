#pragma once

#include "vector_math.h"

#include <iostream>

class Camera
{
public:
	Camera()
		: m_eye(make_float3(0.0f, 0.0f, 0.0f))
		, m_lookat(make_float3(0.0f, 0.0f, 0.0f))
		, m_up(make_float3(0.0f, 0.0f, 0.0f))
		, m_fovY(0.0f)
		, m_aspectRatio(1.0f)
	{ }

	const float3& eye() const { return m_eye; }

	void setEye(const float3& val) { m_eye = val; }
	void setLookat(const float3& val) { m_lookat = val; }
	void setUp(const float3& val) { m_up = val; }
	void setFovY(float val) { m_fovY = val; }
	void setAspectRatio(float val) { m_aspectRatio = val; }

	void setMoveSpeed(const float& val) { m_moveSpeed = val; }

	void UVWFrame(float3& U, float3& V, float3& W) const;

	// Mouse tracking
	void updateTracking(int x, int y);

private:
	float3 m_eye;
	float3 m_lookat;
	float3 m_up;
	float  m_fovY;
	float  m_aspectRatio;

	float m_moveSpeed = 1.0f;

	float3 m_u = make_float3(0.0f, 0.0f, 0.0f);		// Right
	float3 m_v = make_float3(0.0f, 0.0f, 0.0f);		// Up
	float3 m_w = make_float3(0.0f, 0.0f, 0.0f);		// Forward
};