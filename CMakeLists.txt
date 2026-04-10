cmake_minimum_required(VERSION 3.10)

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
    # src/AirCanvas.cpp - I will uncomment it after implementation 
)

# Linking with OpenCV
target_link_libraries(AirCanvas ${OpenCV_LIBS})