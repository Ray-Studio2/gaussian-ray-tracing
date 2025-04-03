# 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes
This repository is an **unofficial implementation** of the paper **"3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes"**. It may not fully reproduce all the results or optimizations presented in the original paper, and it currently contains some bugs that will be continuously addressed in future updates. Additionally, new features will be added over time. Note that training code has not been implemented yet; only a partial application described in the paper is available at this time.

## 🛩️ Features
- **Supported File Format**: Only trained PLY files are supported.   
- **Supported Modes**: Currently reflection and fisheye modes are available.   
- **Gizmo Support**: Includes a gizmo for reflection primitive transform.   
- **OBJ File Loading**: Supports loading OBJ files through a file dialog.

## 🤖 Building gaussian-ray-tracing (Windows)
### Requirements
- Requires an NVIDIA GPU; testing was conducted on an RTX 4090.
- Visual Studio 2022
- __[CUDA](https://developer.nvidia.com/cuda-toolkit)__ 11.4 or higher
- __[CMake](https://cmake.org/)__ v3.8 or higher.
- __[Optix](https://developer.nvidia.com/rtx/ray-tracing/optix)__ 7.7.0 or higher.

### Compilation
Clone repository with submodules.
```sh
$ git clone --recursive https://github.com/GPU-Workers/gaussian-ray-tracing.git
$ cd gaussian-ray-tracing
```

Build the project: (on Windows, use a [developer command prompt](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#developer_command_prompt))
```sh
$ mkdir build
$ cd build

# Add optix directory.
# On windows, the default Optix 7.7 installation path is C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0
$ cmake -DOptiX_INSTALL_DIR="/path/to/optix" ..
$ cmake --build .
```

## 🦤 Examples
### 🪞Reflection
***4090에서 녹화 후 올릴 예정!***
### 🎏 Fisheye
***4090에서 녹화 후 올릴 예정!***

## 🎠 Usage
```sh
$ cd build
$ gaussian-ray-tracing.exe -p /path/to/ply_file --width width_value --height height_value
```

### Keyboard shortcuts
| Key             | Meaning       |
| :-------------: | ------------- |
| WASD            | Forward / Left / Backward / Right. |
| Q / ESC         | Exit. |
| R			      | Reset camera. |
| N               | ON / OFF Rendering normals of reflection primitives. |
| L Ctrl + P / S  | Add reflection primitives. |
| V               | ON / OFF camera mode. (Pinhole / Fisheye) |

## 🙇‍♂️ Acknowledgements
This project was developed with reference to the following repositories and resources:
- [WebGPU Gaussian Tracer](https://github.com/meta-plane/WebGPU-GaussianTracer)
- [NVIDIA OptiX Apps](https://github.com/NVIDIA/OptiX_Apps)
- OptiX SDK examples

## 📒ToDo list
- Gaussian Rendering Issue Improvement   
- Remove Individual Meshes   
- Add Refraction Feature   
- Code Refactoring   
- Miscellaneous Bug Fixes
- Fix the OBJ loader (match the current scene’s scale, resolve normal issues)