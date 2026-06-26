cmake_minimum_required(VERSION 3.10)

project(jetsonApp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(jetsonApp main.cpp)

target_link_libraries(jetsonApp ${OpenCV_LIBS})

