#pragma once

#include <iostream>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#include "vector_math.h"

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
	const float3& lookat() const { return m_lookat; }

	void setEye(const float3& val) { m_eye = val; }
	void setLookat(const float3& val) { m_lookat = val; }
	void setUp(const float3& val) { m_up = val; }
	void setFovY(float val) { m_fovY = val; }
	void setAspectRatio(float val) { m_aspectRatio = val; }
	void setMoveSpeed(const float& val) { m_moveSpeed = val; }

	void UVWFrame(float3& U, float3& V, float3& W) const;

	glm::mat4 getViewMatrix() {
		glm::mat4 view = glm::lookAt(
			glm::vec3(m_eye.x, m_eye.y, m_eye.z),
			glm::vec3(m_lookat.x, m_lookat.y, m_lookat.z),
			glm::vec3(m_up.x, -m_up.y, -m_up.z)
		);

		return view;
	}
	glm::mat4 getProjectionMatrix() {
		glm::mat4 proj = glm::perspective(
			glm::radians(m_fovY),
			m_aspectRatio,
			0.1f,
			100.0f
		);

		return proj;
	}

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