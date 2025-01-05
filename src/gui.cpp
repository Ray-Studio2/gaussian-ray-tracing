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

void GUI::renderPanel(
    std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time
)
{
	ImGui::SetNextWindowPos(ImVec2(860, 20));       // Hardcoded position
	ImGui::SetNextWindowSize(ImVec2(400, 200));	    // Hardcoded size
    ImGui::Begin("Pannel");
    displayText(state_update_time, render_time, display_time);
	ImGui::End();
}

void GUI::displayText(
    std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time
)
{
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
        state_update_time = render_time = display_time = std::chrono::duration<double>::zero();
    }

    ++total_subframe_count;

    ImGui::TextColored(ImColor(0.7f, 0.7f, 0.7f, 1.0f), "%s", display_text);
}