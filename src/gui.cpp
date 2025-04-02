#include "gui.h"
#include "Exception.h"

#include <ImGuiFileDialog.h>

GUI::~GUI()
{
}

GLFWwindow* GUI::initUI(const char* window_title)
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // To make Apple happy -- should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(m_width, m_height, window_title, nullptr, nullptr);
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

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
    io.Fonts->AddFontDefault();

    ImGui::GetStyle().WindowBorderSize = 0.0f;

    return window;
}

void GUI::initCamera(Camera* camera)
{
    camera->setEye(make_float3(0.0f, 0.0f, 3.0f));
    camera->setLookat(center);
    camera->setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera->setFovY(60.0f);

    camera_changed = true;

    m_camera = camera;
    reinitOrientationFromCamera();
    setMoveSpeed(7.0f);
    setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
}

void GUI::resetCamera()
{
    m_camera->setEye(make_float3(0.0f, 0.0f, 3.0f));
    m_camera->setLookat(center);
    m_camera->setUp(make_float3(0.0f, 1.0f, 0.0f));
    m_camera->setFovY(60.0f);

    camera_changed = true;

    reinitOrientationFromCamera();
    setMoveSpeed(7.0f);
    setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    }

void GUI::eventHandler()
{
	mouseEvent();
	keyboardEvent();
}

void GUI::mouseEvent()
{
    if (!ImGui::GetIO().WantCaptureMouse) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver())
        {
            mouse_button = LEFT;
			m_viewMode = EyeFixed;
            startTracking(static_cast<int>(ImGui::GetMousePos().x), static_cast<int>(ImGui::GetMousePos().y));
        }
        else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            mouse_button = RELEASED;
        }
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            mouse_button = RIGHT;
			m_viewMode = LookAtFixed;
            startTracking(static_cast<int>(ImGui::GetMousePos().x), static_cast<int>(ImGui::GetMousePos().y));
        }

        if (mouse_button == LEFT && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            updateTracking(static_cast<int>(ImGui::GetMousePos().x), static_cast<int>(ImGui::GetMousePos().y));

            camera_changed = true;
        }
		else if (mouse_button == RIGHT && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
			updateTracking(static_cast<int>(ImGui::GetMousePos().x), static_cast<int>(ImGui::GetMousePos().y));

			camera_changed = true;
		}

        ImGuiIO& io = ImGui::GetIO();
        float delta = io.MouseWheel;
        if (delta != 0.0f) {
			float3 dir = normalize(m_camera->lookat() - m_camera->eye());
			float3 delta_eye = delta * dir * 0.1f;
			m_camera->setEye(m_camera->eye() + delta_eye);
			m_camera->setLookat(m_camera->lookat() + delta_eye);
			camera_changed = true;
        }
        
        reinitOrientationFromCamera();
    }
}

void GUI::keyboardEvent()
{
    // Exit
    if (ImGui::IsKeyPressed(ImGuiKey_Q) || ImGui::IsKeyPressed(ImGuiKey_Escape))
		glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);

	// Camera movement
	if (ImGui::IsKeyPressed(ImGuiKey_A)) {
        float3 right_move = m_u * 0.01f * m_moveSpeed;
        m_camera->setEye(m_camera->eye() + right_move);
        m_camera->setLookat(m_camera->lookat() + right_move);
        camera_changed = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_D)) {
        float3 left_move = -m_u * 0.01f * m_moveSpeed;
        m_camera->setEye(m_camera->eye() + left_move);
        m_camera->setLookat(m_camera->lookat() + left_move);
        camera_changed = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        float3 forward_move = -m_v * 0.01f * m_moveSpeed;
        m_camera->setEye(m_camera->eye() + forward_move);
        m_camera->setLookat(m_camera->lookat() + forward_move);
        camera_changed = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_S)) {
        float3 back_move = m_v * 0.01f * m_moveSpeed;
        m_camera->setEye(m_camera->eye() + back_move);
        m_camera->setLookat(m_camera->lookat() + back_move);
        camera_changed = true;
    }

 //   // Add primitives
	//if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyPressed(ImGuiKey_P)) {
	//	m_tracer->createGeometry<Plane>(geometries[PLANE]);
	//}
 //   else if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyPressed(ImGuiKey_S)) {
	//	m_tracer->createGeometry<Sphere>(geometries[SPHERE]);
 //   }

    // Render reflection primitive normals
	if (ImGui::IsKeyPressed(ImGuiKey_N)) {
		reflection_render_normals = !reflection_render_normals;
		m_tracer->setReflectionMeshRenderNormal(reflection_render_normals);
    }

	// Set camera mode
	if (ImGui::IsKeyPressed(ImGuiKey_V)) {
		is_fisheye_mode = !is_fisheye_mode;
		m_tracer->params.mode_fisheye = is_fisheye_mode;
	}

	// Reset camera
	if (ImGui::IsKeyPressed(ImGuiKey_R)) {
        resetCamera();
	}
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

    reinitOrientationFromCamera();
}

void GUI::updateCamera()
{
    // use latlon for view definition
    float3 localDir;
    localDir.x = cos(m_latitude) * sin(m_longitude);
    localDir.y = cos(m_latitude) * cos(m_longitude);
    localDir.z = sin(m_latitude);

    float3 dirWS = m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;
    
    if (m_viewMode == EyeFixed) {
        const float3& eye = m_camera->eye();
        m_camera->setLookat(eye - dirWS * m_cameraEyeLookatDistance);
    }
    else {
        const float3& lookat = m_camera->lookat();
        m_camera->setEye(lookat + dirWS * m_cameraEyeLookatDistance);
    }
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
    std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time
    )
{
    displayText(state_update_time, render_time, display_time);
    renderPanel();
}

void GUI::renderPanel()
{
	ImGui::SetNextWindowPos(ImVec2(m_width - 320.0f, 20));
    ImGui::SetNextWindowSize(ImVec2(300, m_height - 40));
    ImGui::Begin("Pannel");

    if (ImGui::CollapsingHeader("DEBUG"))
	{
        ImGui::PushItemWidth(100);

		ImGui::SliderInt("Hit array size", &m_tracer->params.k, 1, 6);
		ImGui::SliderFloat("Alpha min", &m_tracer->params.alpha_min, 0.01f, 0.2f);
		ImGui::SliderFloat("T min", &m_tracer->params.T_min, 0.03f, 0.99f);

        ImGui::PopItemWidth();
	}

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Camera Mode")) {
        if (ImGui::RadioButton("Pinhole", !is_fisheye_mode)) {
            is_fisheye_mode = false;
            m_tracer->params.mode_fisheye = is_fisheye_mode;
        }

        ImGui::SameLine();

        if (ImGui::RadioButton("Fisheye", is_fisheye_mode)) {
            is_fisheye_mode = true;
            m_tracer->params.mode_fisheye = is_fisheye_mode;
        }
    }

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Reflection"))
    {
		if (ImGui::Button("Add Primitive"))
		{
            if (selected_geometry == PLANE) {
                m_tracer->createPlane();
            }
            else if (selected_geometry == SPHERE) {
                m_tracer->createSphere();
            }
            else if (selected_geometry == CUSTOM) {
				open_file_dialog = true;
            }
		}

        if (open_file_dialog) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", config);
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

				m_tracer->createLoadMesh(filePathName);
            }
            ImGuiFileDialog::Instance()->Close();

            open_file_dialog = false;
        }

        ImGui::SameLine();

		ImGui::PushItemWidth(70);
		ImGui::Combo("Primitive Type", &selected_geometry, geometries, IM_ARRAYSIZE(geometries));
		ImGui::PopItemWidth();

        ImGui::Spacing();

        if (ImGui::Checkbox("Render Normals", &reflection_render_normals)) {
            m_tracer->setReflectionMeshRenderNormal(reflection_render_normals);
        }

        ImGuizmo::BeginFrame();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
        ImGuizmo::SetRect(0, 0, m_width, m_height);

        glm::mat4 manipulated_model;
        int manipulated_index = -1;

        for (Primitive& p : m_tracer->getPrimitives()) {
            std::string lbl = p.type + " " + std::to_string(p.index);
            int node_id = p.index;

            if (close_node == node_id) {
                ImGui::SetNextItemOpen(false, ImGuiCond_Always);
                close_node = -1;
            }

            ImGui::PushID(node_id);
            if (ImGui::TreeNode(lbl.c_str())) {
                if (current_node == node_id) {
                    if (ImGui::RadioButton("Translation", m_currentGizmoOperation == ImGuizmo::TRANSLATE))
                        m_currentGizmoOperation = ImGuizmo::TRANSLATE;

                    if (ImGui::RadioButton("Rotation", m_currentGizmoOperation == ImGuizmo::ROTATE))
                        m_currentGizmoOperation = ImGuizmo::ROTATE;

                    if (ImGui::RadioButton("Scale", m_currentGizmoOperation == ImGuizmo::SCALE))
                        m_currentGizmoOperation = ImGuizmo::SCALE;

                    manipulated_model = p.transform;
                    manipulated_index = node_id;
                }
                else {
                    close_node = current_node;
                    current_node = node_id;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        if (manipulated_index >= 0) {
            glm::mat4 view = m_camera->getViewMatrix();
            glm::mat4 proj = m_camera->getProjectionMatrix();

            bool manipulated = ImGuizmo::Manipulate(
                glm::value_ptr(view),
                glm::value_ptr(proj),
                m_currentGizmoOperation,
                m_currentGizmoMode,
                glm::value_ptr(manipulated_model),
                nullptr,
                nullptr
            );

            if (manipulated) {
                for (Primitive& p : m_tracer->getPrimitives()) {
                    if (p.index == manipulated_index) {
                        p.transform = manipulated_model;
                        m_tracer->updateInstanceTransforms(p);
                        break;
                    }
                }
            }
        }
             
    //    for (Primitive& p : m_tracer->getPrimitives())
    //    {
    //        std::string lbl = p.type + " " + std::to_string(p.index);
    //        int node_id = p.index;

    //        if (close_node == node_id) {
    //            ImGui::SetNextItemOpen(false, ImGuiCond_Always);
    //            close_node = -1;
    //        }

    //        ImGui::PushID(node_id);
    //        if (ImGui::TreeNode(lbl.c_str())) {
    //            if (current_node == node_id) {
    //                if (ImGui::RadioButton("Translation", m_currentGizmoOperation == ImGuizmo::TRANSLATE))
    //                    m_currentGizmoOperation = ImGuizmo::TRANSLATE;

    //                if (ImGui::RadioButton("Rotation", m_currentGizmoOperation == ImGuizmo::ROTATE))
    //                    m_currentGizmoOperation = ImGuizmo::ROTATE;

    //                if (ImGui::RadioButton("Scale", m_currentGizmoOperation == ImGuizmo::SCALE))
    //                    m_currentGizmoOperation = ImGuizmo::SCALE;

    //                manipulated_model = p.transform;
    //                manipulated_index = node_id;

    //                // Remove primitive
    //                if (ImGui::Button("Remove")) {
    //                    remove_primitive_type = p.type;
    //                    remove_primitive_index = p.index;
    //                    remove_instance_index = p.instanceIndex;
    //                    remove_primitive = true;

    //                    if (current_node == node_id)
    //                        current_node = -1;
    //                }
    //            }
    //            else {
    //                close_node = current_node;
    //                current_node = node_id;
    //            }
    //            ImGui::TreePop();
    //        }
    //        ImGui::PopID();
    //    }

    //    if (manipulated_index >= 0) {
    //        glm::mat4 view = m_camera->getViewMatrix();
    //        glm::mat4 proj = m_camera->getProjectionMatrix();

    //        bool manipulated = ImGuizmo::Manipulate(
    //            glm::value_ptr(view),
    //            glm::value_ptr(proj),
    //            m_currentGizmoOperation,
    //            m_currentGizmoMode,
    //            glm::value_ptr(manipulated_model),
    //            nullptr,
    //            nullptr
    //        );

    //        if (manipulated) {
    //            for (Primitive& p : m_tracer->getPrimitives()) {
    //                if (p.index == manipulated_index) {
    //                    p.transform = manipulated_model;
    //                    m_tracer->updateInstanceTransforms(p);
    //                    break;
    //                }
    //            }
    //        }
    //    }

    //    if (remove_primitive)
    //    {
    //        m_tracer->removePrimitive(remove_primitive_type, remove_primitive_index, remove_instance_index);
    //        remove_primitive = false;
    //    }
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