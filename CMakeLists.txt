﻿cmake_minimum_required(VERSION 3.8)

project ("gaussian-tracing")

set(CMAKE_MODULE_PATH
	${CMAKE_SOURCE_DIR}/CMake
	${CMAKE_MODULE_PATH}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

# Happly
# happly.h max conflicts with windows.h
add_definitions(-DNOMINMAX)

# CUDA
find_package(CUDA 5.0 REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Optix
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})

# OpenGL
find_package(OpenGL REQUIRED)

# GLFW
set(GLFW_DIR third_party/GLFW)
add_subdirectory(${GLFW_DIR})

# IMGUI
set(IMGUI
	third_party/imgui/imgui.cpp
	third_party/imgui/imgui_draw.cpp
	third_party/imgui/imgui_widgets.cpp
	third_party/imgui/backends/imgui_impl_glfw.cpp
	third_party/imgui/backends/imgui_impl_opengl3.cpp
	third_party/imgui/imgui_demo.cpp
	third_party/imgui/imgui_tables.cpp
	third_party/ImGuizmo/ImGuizmo.cpp
)

include_directories(third_party/happly)
include_directories(third_party/glm)
include_directories(third_party/imgui/backends)
include_directories(third_party/imgui)
include_directories(${GLFW_DIR}/include)
include_directories(third_party/glad)
include_directories(third_party)

include_directories(third_party/ImGuiFileDialog)
include_directories(third_party/tinyobjloader)

set(SHADERS
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/tracer.cu
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/tracer.cuh
	${CMAKE_CURRENT_SOURCE_DIR}/src/Parameters.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/vector_math.h
)

CUDA_WRAP_SRCS(
	gaussian-tracing
	PTX
	ptx_files
	${SHADERS}
)

add_library( glad SHARED
  third_party/KHR/khrplatform.h
  third_party/glad/glad.c
  third_party/glad/glad.h
)

target_compile_definitions( glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC third_party/glad )
target_link_libraries( glad PUBLIC ${OPENGL_LIBRARIES} )

add_executable(
	gaussian-tracing
	src/main.cpp
	src/GaussianTracer.cpp
	src/GaussianData.cpp
	src/Exception.h
	src/gui.cpp
	src/Parameters.h
	src/Display.h
	src/Display.cpp
	src/Camera.h
	src/Camera.cpp
	src/vector_math.h
	src/CUDAOutputBuffer.h
	src/CUDAOutputBuffer.cpp
	src/Utility.h
	src/geometry/Icosahedron.h
	src/geometry/Primitives.cpp
	${IMGUI}
	third_party/ImGuiFileDialog/ImGuiFileDialog.cpp
	shaders/tracer.cu
	shaders/tracer.cuh
 )

target_compile_definitions(gaussian-tracing PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(gaussian-tracing glfw glad ${OPENGL_gl_LIBRARY})

target_link_libraries(
	gaussian-tracing
	${OptiX_LIBRARIES}
	${CUDA_LIBRARIES}
)
