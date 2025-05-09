#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/type_ptr.hpp>

#include "ImGuizmo/ImGuizmo.h"

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
	GUI(GaussianTracer* tracer) { m_tracer = tracer; }
	~GUI();

	GLFWwindow* initUI(const char* window_title);
	void beginFrame();
	void endFrame();
	void renderGUI(
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	void setSize(int width, int height) { m_width = width; m_height = height; }

	void setGaussianCenter(float3 gs_center) { center = gs_center; }

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
	void renderPanel();
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
	void resetCamera();

	// Helper functions
	float radians(float degrees) { return degrees * M_PIf / 180.0f; }
	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	GaussianTracer* m_tracer = nullptr;

	// GUI variables
	int m_width  = 0;
	int m_height = 0;

	float3 center = make_float3(0.0f, 0.0f, 0.0f);

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
	const char* geometries[3] = { "Plane", "Sphere", "Load" };
	int			selected_geometry = 0;

	bool open_file_dialog = false;

	// ImGuizmo
	ImGuizmo::OPERATION m_currentGizmoOperation = ImGuizmo::TRANSLATE;
	ImGuizmo::MODE		m_currentGizmoMode = ImGuizmo::LOCAL;

	int close_node   = -1;
	int current_node = -1;

	// Camera mode
	bool is_fisheye_mode = false;

	// Render type flag
	bool renderMirror = true;
	bool renderNormal = false;
	bool renderGlass  = false;
};