# 3D Gaussian Ray Tracing
Implementation of 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes

## Building gaussian-ray-tracing (Windows)
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
# On windows, the default installation path is C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0
$ cmake -DOptiX_INSTALL_DIR="/path/to/optix" ..
$ cmake --build .
```

## Usage
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