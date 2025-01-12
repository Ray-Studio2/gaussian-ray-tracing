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
};