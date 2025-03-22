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
	~Mesh() {};

	std::vector<float3>& getVertices() { return vertices; }
	std::vector<float3>& getNormals() { return normals; }
	std::vector<unsigned int>& getIndices() { return indices; }

	float3 getPosition() const { return position; }
	float3 getRotation() const { return rotation; }		// Degrees
	float3 getScale()    const { return scale; }

	glm::mat4 getTransform() const {
		float tx = position.x;
		float ty = position.y;
		float tz = position.z;

		float yaw = rotation.x;
		float pitch = rotation.y;
		float roll = rotation.z;

		glm::mat4 translation_mat = glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, tz));

		glm::mat4 Ryaw = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 Rpitch = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 Rroll = glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 rotation_mat = Rroll * Rpitch * Ryaw;

		glm::mat4 transform = translation_mat * rotation_mat;

		return transform;
	}

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

	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	std::vector<float3>		  vertices = {};
	std::vector<float3>		  normals = {};
	std::vector<unsigned int> indices = {};

	float3 position = make_float3(0.0f, 0.0f, 0.0f);
	float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
	float3 scale = make_float3(1.0f, 1.0f, 1.0f);
};