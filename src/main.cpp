#include "gui.h"
#include "CUDAOutputBuffer.h"
#include "GaussianTracer.h"
#include "Display.h"
#include "Camera.h"

GUI		gui;
Camera  camera;
int32_t mouse_button = -1;
bool    camera_changed = true;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS)
	{
		mouse_button = button;
		gui.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else
	{
		mouse_button = -1;
	}
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
	{
		gui.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos));
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		// TODO: Implement Eye Fixed mode
	}
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}
	}
}

void initCamera()
{
	camera.setEye(make_float3(0.0f, 0.0f, 3.0f));
	camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
	camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	camera.setFovY(60.0f);

	camera_changed = true;

	gui.setCamera(&camera);
	gui.setMoveSpeed(10.0f);
	gui.setReferenceFrame(
		make_float3(1.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 1.0f),
		make_float3(0.0f, 1.0f, 0.0f)
	);
}

int main() {
	const std::string filename = "../../data/train.ply";
	GaussianTracer tracer(filename);
	
	unsigned int width  = 1280;
	unsigned int height = 720;

	tracer.setSize(width, height);
	tracer.initializeOptix();

	GLFWwindow* window = gui.initUI("Gaussian Tracer", width, height);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetKeyCallback(window, keyCallback);

	GLDisplay gldisplay;

	CUDAOutputBuffer output_buffer(width, height);
	output_buffer.setStream(tracer.stream);

	initCamera();

	//gui.setCamera(&camera);

	//tracer.setCamera(camera);
	//tracer.initCamera();
	tracer.initParams();

	std::chrono::duration<double> state_update_time(0.0);
	std::chrono::duration<double> render_time(0.0);
	std::chrono::duration<double> display_time(0.0);

	while (!glfwWindowShouldClose(window))
	{
		auto t0 = std::chrono::steady_clock::now();

		glfwPollEvents();

		tracer.updateCamera(camera, camera_changed);

		auto t1 = std::chrono::steady_clock::now();
		state_update_time += t1 - t0;
		t0 = t1;

		tracer.render(output_buffer);
		t1 = std::chrono::steady_clock::now();
		render_time += t1 - t0;
		t0 = t1;

		int framebuf_res_x = 0;
		int framebuf_res_y = 0;
		glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);

		gldisplay.display(
			output_buffer.width(),
			output_buffer.height(),
			framebuf_res_x,
			framebuf_res_y,
			output_buffer.getPBO()
		);
		t1 = std::chrono::steady_clock::now();
		display_time += t1 - t0;
		t0 = t1;

		gui.beginFrame();
		gui.renderGUI(tracer.params, state_update_time, render_time, display_time);
		gui.endFrame();

		glfwSwapBuffers(window);
	}

	glfwTerminate();

	return 0;
}