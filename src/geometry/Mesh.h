#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>
#include <random>

#include "../vector_math.h"

class Mesh
{
public:
	Mesh() {}
	Mesh(float3 center) {}
	Mesh(float3 center, std::string filename) {}
	~Mesh() {};

	std::vector<float3>& getVertices() { return vertices; }
	std::vector<float3>& getNormals() { return normals; }
	std::vector<unsigned int>& getIndices() { return indices; }

	glm::mat4 getTransform() const { return m_transform; }

protected:
	virtual void createGeometry() = 0;

	void setPosition(float3 center)
	{
		// Fixed position to gaussian partices center.
		float tx = center.x;
		float ty = center.y;
		float tz = center.z;

		position = make_float3(tx, ty, tz);
	}

	void setInitialTransform()
	{
		glm::mat4 t       = glm::translate(glm::mat4(1.0f), glm::vec3(position.x, position.y, position.z));
		glm::mat4 r_yaw   = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 r_pitch = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 r_roll  = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));\
		glm::mat4 r		  = r_yaw * r_pitch * r_roll;
		glm::mat4 s       = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

		m_transform = t * r * s;
	}

	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	std::vector<float3>		  vertices = {};
	std::vector<float3>		  normals = {};
	std::vector<unsigned int> indices = {};

	float3 position = make_float3(0.0f, 0.0f, 0.0f);
	glm::mat4 m_transform = glm::mat4(1.0f);
};