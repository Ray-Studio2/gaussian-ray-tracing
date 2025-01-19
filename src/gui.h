#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    include<windows.h>
#    include<mmsystem.h>
#else
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>

#endif

#include <chrono>
#include <iostream>

#include "GaussianTracer.h"
#include "Camera.h"

class GUI
{
public:
	GUI();
	~GUI();

	GLFWwindow* initUI(const char* window_title, int width, int height);
	void beginFrame();
	void endFrame();
	void renderGUI(
		GaussianTracer& tracer,
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	inline void setCamera(Camera* camera) { m_camera = camera; reinitOrientationFromCamera(); }

	void setMoveSpeed(const float& val) { m_moveSpeed = val; }
	void reinitOrientationFromCamera();
	void setReferenceFrame(const float3& u, const float3& v, const float3& w);

	// Mouse tracking
	void startTracking(int x, int y);
	void updateTracking(int x, int y);

private:
	static void errorCallback(int error, const char* description)
	{
		std::cerr << "GLFW Error " << error << ": " << description << std::endl;
	}

	void renderPanel(GaussianTracer& tracer);
	void displayText(
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	float radians(float degrees) { return degrees * M_PIf / 180.0f; }
	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	void updateCamera();

	Camera* m_camera = nullptr;

	float m_moveSpeed = 1.0f;
	float m_cameraEyeLookatDistance = 0.0f;

	int  m_prevPosX = 0;
	int  m_prevPosY = 0;
	bool m_tracking = false;

	float m_latitude  = 0.0f;   // in radians
	float m_longitude = 0.0f;   // in radians

	float3 m_u = make_float3(0.0f, 0.0f, 0.0f);
	float3 m_v = make_float3(0.0f, 0.0f, 0.0f);
	float3 m_w = make_float3(0.0f, 0.0f, 0.0f);

	// Primitives
	const char* geometries[2] = {"Sphere", "Plane"};
	int			selected_geometry = 0;
};