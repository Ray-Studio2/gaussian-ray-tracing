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

enum MouseButton
{
	LEFT = 0,
	RIGHT,
	MIDDLE,
	RELEASED
};

enum ViewMode
{
	EyeFixed = 0,
	LookAtFixed
};

class GUI
{
public:
	GUI();
	~GUI();

	GLFWwindow* initUI(const char* window_title, int width, int height);
	void beginFrame();
	void endFrame();
	void renderGUI(
		GaussianTracer* tracer,
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	// Camera
	void initCamera(Camera* camera);

	// Event handling
	void eventHandler();

	// Camera
	Camera* m_camera    = nullptr;
	bool camera_changed = false;

private:
	static void errorCallback(int error, const char* description)
	{
		std::cerr << "GLFW Error " << error << ": " << description << std::endl;
	}

	// GUI functions
	void renderPanel(GaussianTracer* tracer);
	void displayText(
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	// Event handling functions
	void mouseEvent();
	void keyboardEvent();
	
	// Camera functions
	void setMoveSpeed(const float& val) { m_moveSpeed = val; }
	void reinitOrientationFromCamera();
	void setReferenceFrame(const float3& u, const float3& v, const float3& w);
	void startTracking(int x, int y);
	void updateTracking(int x, int y);
	void updateCamera();

	// Helper functions
	float radians(float degrees) { return degrees * M_PIf / 180.0f; }
	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	// Camera variables
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

	ViewMode m_viewMode = EyeFixed;

	// Event variables
	MouseButton mouse_button = RELEASED;

	// Primitives
	const char* geometries[2] = { "Plane", "Sphere" };
	int			selected_geometry = 0;

	// Transform flags
	bool updated_tx          = false;
	bool updated_ty          = false;
	bool updated_tz          = false;
	bool updated_yaw         = false;
	bool updated_pitch       = false;
	bool updated_roll        = false;
	bool updated_sx          = false;
	bool updated_sy          = false;
	bool updated_sz          = false;
	bool updated_translation = false;
	bool updated_rotation    = false;
	bool updated_scale       = false;

	// Remove flag
	bool remove_primitive             = false;
	std::string remove_primitive_type = "";
	size_t remove_instance_index      = 0;
	size_t remove_primitive_index     = 0;
};