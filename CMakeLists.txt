cmake_minimum_required(VERSION 3.10)

# enable packages download module
include(FetchContent)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)

# download toml++ and make available
FetchContent_MakeAvailable(tomlplusplus)

project(AirCanvasCV)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# OpenCV header files
include_directories(${OpenCV_INCLUDE_DIRS})

# src header files
include_directories(src)

add_executable(AirCanvas 
    src/main.cpp 
    src/AirCanvas.cpp 
)

# Linking with OpenCV and toml++
target_link_libraries(AirCanvas ${OpenCV_LIBS} tomlplusplus::tomlplusplus)