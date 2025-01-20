#include "gui.h"
#include "Exception.h"

GUI::GUI()
{
}

GUI::~GUI()
{
}

GLFWwindow* GUI::initUI(const char* window_title, int width, int height)
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // To make Apple happy -- should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, window_title, nullptr, nullptr);
    if (!window)
        throw std::runtime_error("Failed to create GLFW window");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // No vsync

    if (!gladLoadGL())
        throw std::runtime_error("Failed to initialize GL");

    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
    io.Fonts->AddFontDefault();

    ImGui::GetStyle().WindowBorderSize = 0.0f;

    return window;
}

void GUI::reinitOrientationFromCamera()
{
    m_camera->UVWFrame(m_u, m_v, m_w);
    m_u = normalize(m_u);
    m_v = normalize(m_v);
    m_w = normalize(-m_w);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_cameraEyeLookatDistance = length(m_camera->lookat() - m_camera->eye());
}

void GUI::startTracking(int x, int y)
{
    m_prevPosX = x;
	m_prevPosY = y;
	m_tracking = true;
}

void GUI::updateTracking(int x, int y)
{
    if (!m_tracking) {
		startTracking(x, y);
		return;
    }
	
	int deltaX = -(x - m_prevPosX);
    int deltaY = -(y - m_prevPosY);

    m_prevPosX = x;
    m_prevPosY = y;
    m_latitude  = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * deltaY)));
    m_longitude = radians(fmod(degrees(m_longitude) - 0.5f * deltaX, 360.0f));

    updateCamera();
}

void GUI::updateCamera()
{
    // use latlon for view definition
    float3 localDir;
    localDir.x = cos(m_latitude) * sin(m_longitude);
    localDir.y = cos(m_latitude) * cos(m_longitude);
    localDir.z = sin(m_latitude);

    float3 dirWS = m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;
    
    const float3& eye = m_camera->eye();
    m_camera->setLookat(eye - dirWS * m_cameraEyeLookatDistance);
}

void GUI::setReferenceFrame(const float3& u, const float3& v, const float3& w)
{
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -normalize(m_camera->lookat() - m_camera->eye());
    float3 dirLocal;
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void GUI::beginFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::endFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::renderGUI(
    GaussianTracer& tracer,
    std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time
    )
{
    displayText(state_update_time, render_time, display_time);
    renderPanel(tracer);
}

void GUI::renderPanel(GaussianTracer& tracer)
{
	ImGui::SetNextWindowPos(ImVec2(960, 20));
	ImGui::SetNextWindowSize(ImVec2(300, 680));
    ImGui::Begin("Pannel");

    if (ImGui::CollapsingHeader("DEBUG"))
	{
        ImGui::PushItemWidth(100);

		ImGui::SliderInt("Hit array size", &tracer.params.k, 1, 6);
		ImGui::SliderFloat("Alpha min", &tracer.params.alpha_min, 0.01f, 0.2f);
		ImGui::SliderFloat("T min", &tracer.params.T_min, 0.03f, 0.99f);
        ImGui::Checkbox("Visualize hit count", &tracer.params.visualize_hitcount);

        ImGui::PopItemWidth();
	}

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Reflection"))
    {
		if (ImGui::Button("Add Primitive"))
		{
            if (selected_geometry == 0) {
                tracer.createPlane();
            }
            else if (selected_geometry == 1) {
                //tracer.addSphere();
            }
		}
		
        ImGui::SameLine();

		ImGui::PushItemWidth(70);
		ImGui::Combo("Primitive Type", &selected_geometry, geometries, IM_ARRAYSIZE(geometries));
		ImGui::PopItemWidth();

		for (Primitive& p : tracer.getPrimitives())
        {
			std::string lbl = p.type + " " + std::to_string(p.index);
            if (ImGui::TreeNode(lbl.c_str()))
            {
				ImGui::SliderFloat3("Translate", &p.position.x, -1.0f, 1.0f, "%.2f", 0.01f);
				ImGui::SliderFloat3("Rotate", &p.rotation.x, -180.0f, 180.0f, "%.2f", 3.6f);
				ImGui::SliderFloat3("Scale", &p.scale.x, 0.1f, 2.0f, "%.2f", 0.01f);

				/*tracer.updateInstanceTransforms(p);*/

				ImGui::TreePop();
			}
            tracer.updateInstanceTransforms(p);
		}   
    }

	ImGui::End();
}

void GUI::displayText(
    std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time
    )
{
    ImGui::SetNextWindowPos(ImVec2(20, 20));
    ImGui::SetNextWindowSize(ImVec2(200, 80));
    ImGui::Begin("FPS", 0, ImGuiWindowFlags_NoDecoration);

    constexpr std::chrono::duration<double> display_update_min_interval_time(0.5);
    static int32_t                          total_subframe_count = 0;
    static int32_t                          last_update_frames = 0;
    static auto                             last_update_time = std::chrono::steady_clock::now();
    static char                             display_text[128];

    const auto cur_time = std::chrono::steady_clock::now();

    last_update_frames++;

    typedef std::chrono::duration<double, std::milli> durationMs;

    if (cur_time - last_update_time > display_update_min_interval_time || total_subframe_count == 0)
    {
        sprintf(display_text,
            "%5.1f fps\n\n"
            "state update: %8.1f ms\n"
            "render      : %8.1f ms\n"
            "display     : %8.1f ms\n",
            last_update_frames / std::chrono::duration<double>(cur_time - last_update_time).count(),
            (durationMs(state_update_time) / last_update_frames).count(),
            (durationMs(render_time) / last_update_frames).count(),
            (durationMs(display_time) / last_update_frames).count());

        last_update_time = cur_time;
        last_update_frames = 0;

        state_update_time = std::chrono::duration<double>::zero();
        render_time       = std::chrono::duration<double>::zero();
        display_time      = std::chrono::duration<double>::zero();
    }

    ++total_subframe_count;

    ImGui::TextColored(ImColor(0.7f, 0.7f, 0.7f, 1.0f), "%s", display_text);

    ImGui::End();
}