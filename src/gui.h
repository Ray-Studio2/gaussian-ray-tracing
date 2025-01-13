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

#include "Parameters.h"
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
		Params& params,
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	inline void setCamera(Camera* camera) { m_camera = camera; }

	// Mouse tracking
	void startTracking(int x, int y);
	void updateTracking(int x, int y);

private:
	static void errorCallback(int error, const char* description)
	{
		std::cerr << "GLFW Error " << error << ": " << description << std::endl;
	}

	void renderPanel(Params& params);
	void displayText(
		std::chrono::duration<double>& state_update_time,
		std::chrono::duration<double>& render_time,
		std::chrono::duration<double>& display_time
	);

	float radians(float degrees) { return degrees * M_PIf / 180.0f; }
	float degrees(float radians) { return radians * 180.0f / M_PIf; }

	Camera* m_camera = nullptr;

	int  m_prevPosX = 0;
	int  m_prevPosY = 0;
	bool m_tracking = false;

	float m_latitude  = 0.0f;   // in radians
	float m_longitude = 0.0f;   // in radians
};