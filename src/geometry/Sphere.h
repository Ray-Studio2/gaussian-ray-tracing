#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>
#include <random>

#include "../vector_math.h"

class Sphere
{
public:
    Sphere()
    {
        createSphere();
        setPosition();
        setRotation();
        setScale();
    }
    ~Sphere() {}

    std::vector<float3>& getVertices() { return vertices; }
    std::vector<unsigned int>& getIndices() { return indices; }

    float3 getPosition() const { return position; }
    float3 getRotation() const { return rotation; }
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

    void createSphere()
    {
        // TODO: Hardcoded..
        unsigned int tessU = 180;
        unsigned int tessV = 90;
        float        radius = 0.3f;
        float        maxTheta = M_PIf;

        vertices.reserve((tessU + 1) * tessV);
        indices.reserve(6 * tessU * (tessV - 1));

        float phi_step = 2.0f * M_PIf / (float)tessU;
        float theta_step = maxTheta / (float)(tessV - 1);

        // Latitudinal rings.
        // Starting at the south pole going upwards on the y-axis.
        for (unsigned int latitude = 0; latitude < tessV; ++latitude) // theta angle
        {
            float theta = (float)latitude * theta_step;
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);

            float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]

            // Generate vertices along the latitudinal rings.
            // On each latitude there are tessU + 1 vertices.
            // The last one and the first one are on identical positions, but have different texture coordinates!
            // Note that each second triangle connected to the two poles has zero area if the sphere is closed.
            // But since this is also used for open spheres (maxTheta < 1.0f) this is required.
            for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
            {
                float phi = (float)longitude * phi_step;
                float sinPhi = sinf(phi);
                float cosPhi = cosf(phi);

                float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]

                // Unit sphere coordinates are the normals.
                float3 normal = make_float3(cosPhi * sinTheta,
                    -cosTheta,           // -y to start at the south pole.
                    -sinPhi * sinTheta);
                float3 vertex;
                vertex = normal * radius;
                vertices.push_back(vertex);
				normals.push_back(normal);
            }
        }

        // We have generated tessU + 1 vertices per latitude.
        const unsigned int columns = tessU + 1;

        // Calculate indices.
        for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
        {
            for (unsigned int longitude = 0; longitude < tessU; ++longitude)
            {
                indices.push_back(latitude * columns + longitude);  // lower left
                indices.push_back(latitude * columns + longitude + 1);  // lower right
                indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 

                indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
                indices.push_back((latitude + 1) * columns + longitude);  // upper left
                indices.push_back(latitude * columns + longitude);  // lower left
            }
        }
    }

    void setPosition()
    {
        position = make_float3(randomPosition(gen), randomPosition(gen), randomPosition(gen));

        // Fixed position and rotation
        float tx = 1.0f;
        float ty = 0.0f;
        float tz = 2.0f;

        position = make_float3(tx, ty, tz);
    }

    // Degrees
    void setRotation()
    {

        float rot_x = glm::degrees(randomAngle(gen));
        float rot_y = glm::degrees(randomAngle(gen));
        float rot_z = glm::degrees(randomAngle(gen));

        rotation = make_float3(rot_x, rot_y, rot_z);
    }

    void setScale()
    {
        scale = make_float3(1.0f, 1.0f, 1.0f);
    }

    std::vector<float3>       vertices = {};
    std::vector<unsigned int> indices = {};
    std::vector<float3>       normals = {};

    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
};