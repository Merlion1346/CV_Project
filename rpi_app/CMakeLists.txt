cmake_minimum_required(VERSION 3.16)
project(HeadPoseRPi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_AUTOMOC ON)

find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# ONNX Runtime — set -DONNXRUNTIME_DIR=/path/to/onnxruntime if not in system paths
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root (contains include/ and lib/)")
if(ONNXRUNTIME_DIR)
    set(ORT_INCLUDE "${ONNXRUNTIME_DIR}/include")
    find_library(ORT_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
else()
    find_path(ORT_INCLUDE onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime
        PATHS /usr/local/include /usr/include)
    find_library(ORT_LIB onnxruntime PATHS /usr/local/lib /usr/lib REQUIRED)
endif()

message(STATUS "ORT include: ${ORT_INCLUDE}")
message(STATUS "ORT lib:     ${ORT_LIB}")

add_executable(HeadPoseRPi
    main.cpp
    MainWindow.cpp
    HeadPosePredictor.cpp
    CameraThread.cpp
)

target_include_directories(HeadPoseRPi PRIVATE
    ${ORT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(HeadPoseRPi PRIVATE
    Qt5::Widgets
    ${OpenCV_LIBS}
    ${ORT_LIB}
)

install(TARGETS HeadPoseRPi DESTINATION bin)
