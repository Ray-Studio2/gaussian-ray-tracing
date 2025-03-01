#include "gui.h"
#include "CUDAOutputBuffer.h"
#include "GaussianTracer.h"
#include "Display.h"
#include "Camera.h"

int main() 
{
	const std::string filename = "../../data/test.ply";
	GaussianTracer tracer(filename);
	
	unsigned int width  = 1280;
	unsigned int height = 720;

	tracer.setSize(width, height);
	tracer.initializeOptix();

	GUI gui;
	GLFWwindow* window = gui.initUI("Gaussian Tracer", width, height);

	GLDisplay gldisplay;

	CUDAOutputBuffer output_buffer(width, height);
	output_buffer.setStream(tracer.stream);

	Camera camera;
	gui.initCamera(&camera);
	tracer.initParams();

	std::chrono::duration<double> state_update_time(0.0);
	std::chrono::duration<double> render_time(0.0);
	std::chrono::duration<double> display_time(0.0);

	while (!glfwWindowShouldClose(window))
	{
		auto t0 = std::chrono::steady_clock::now();

		glfwPollEvents();

		gui.eventHandler();

		tracer.updateCamera(camera, gui.camera_changed);

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
		gui.renderGUI(&tracer, state_update_time, render_time, display_time);
		gui.endFrame();

		glfwSwapBuffers(window);
	}

	glfwTerminate();

	return 0;
}