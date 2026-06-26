cmake_minimum_required(VERSION 3.10)

project(jetsonApp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(jetsonApp main.cpp)

