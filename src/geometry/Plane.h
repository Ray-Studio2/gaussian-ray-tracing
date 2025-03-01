#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>
#include <random>

#include "../vector_math.h"

class Plane
{
public:
	Plane()
	{
		createPlane();
		setPosition();
		setRotation();
		setScale();
	}
	~Plane() {};

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

private:
	std::mt19937 gen{ std::random_device{}() };

	std::uniform_real_distribution<float> randomPosition{ -0.5f, 0.5f };
	std::uniform_real_distribution<float> randomAngle{ glm::radians(-30.0f), glm::radians(30.0f) };

	void createPlane()
	{
		unsigned int tessU = 1;
		unsigned int tessV = 1;

		const float width = 0.3f;
		const float height = 0.5f;

		const float uTile = width / float(tessU);
		const float vTile = height / float(tessV);

		float3 corner = make_float3(-width * 0.5f, -height * 0.5f, 0.0f);
		float3 normal = make_float3(0.0f, 0.0f, -1.0f);

		for (unsigned int j = 0; j <= tessV; ++j)
		{
			const float v = float(j) * vTile;

			for (unsigned int i = 0; i <= tessU; ++i)
			{
				const float u = float(i) * uTile;

				float3 vertex = corner + make_float3(u, v, 0.0f);

				vertices.push_back(vertex);
				normals.push_back(normal);
			}
		}

		const unsigned int stride = tessU + 1;
		for (unsigned int j = 0; j < tessV; ++j)
		{
			for (unsigned int i = 0; i < tessU; ++i)
			{
				indices.push_back(j * stride + i);
				indices.push_back(j * stride + i + 1);
				indices.push_back((j + 1) * stride + i + 1);

				indices.push_back((j + 1) * stride + i + 1);
				indices.push_back((j + 1) * stride + i);
				indices.push_back(j * stride + i);
			}
		}
	}

	void setPosition()
	{
		position = make_float3(randomPosition(gen), randomPosition(gen), randomPosition(gen));

		// Fixed position and rotation
		float tx = 0.0f;
		float ty = 0.0f;
		float tz = 5.0f;

		position = make_float3(tx, ty, tz);
	}

	// Degrees
	void setRotation()
	{
		float rot_x = glm::degrees(randomAngle(gen));
		float rot_y = glm::degrees(randomAngle(gen));
		float rot_z = glm::degrees(randomAngle(gen));

		rotation = make_float3(rot_x, rot_y, rot_z);

		rot_x = 0.0f;
		rot_y = 0.0f;
		rot_z = 0.0f;

		rotation = make_float3(rot_x, rot_y, rot_z);
	}

	void setScale()
	{
		scale = make_float3(1.0f, 1.0f, 1.0f);
	}

	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	std::vector<float3>		  vertices = {};
	std::vector<float3>		  normals = {};
	std::vector<unsigned int> indices = {};

	float3 position = make_float3(0.0f, 0.0f, 0.0f);
	float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
	float3 scale = make_float3(1.0f, 1.0f, 1.0f);
};
