#include "gui.h"
#include "CUDAOutputBuffer.h"
#include "GaussianTracer.h"
#include "Display.h"
#include "Camera.h"

#include <args/args.hxx>

int main(int argc, char* argv[])
{
	args::ArgumentParser parser(
		"3D Gaussian Ray Tracing\n",
		"Version v1.0.0"
	);

	args::HelpFlag help(
		parser, 
		"HELP",
		"Display this help menu", 
		{ 'h', "help" }
	);

	args::ValueFlag<std::string> ply_flag{
		parser,
		"PLY",
		"The PLY file to load.",
		{'p', "ply"},
	};

	args::ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	args::ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	try {
		parser.ParseCLI(argc, argv);
	}
	catch (const args::Help&) {
		std::cout << parser;
		return 0;
	}
	catch (const args::ParseError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -1;
	}
	catch (const args::ValidationError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -2;
	}

	std::string filename = ply_flag ? args::get(ply_flag) : "../data/test.ply";
	GaussianTracer tracer(filename);
	
	unsigned int width  = width_flag ? args::get(width_flag) : 1280;
	unsigned int height = height_flag ? args::get(height_flag) : 720;

	tracer.setSize(width, height);
	tracer.initializeOptix();

	GUI gui;
	gui.setSize(width, height);
	GLFWwindow* window = gui.initUI("Gaussian Tracer");

	GLDisplay gldisplay;

	CUDAOutputBuffer output_buffer(width, height);
	output_buffer.setStream(tracer.stream);

	Camera camera;
	gui.setGaussianCenter(tracer.getGaussianCenter());
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