cmake_minimum_required (VERSION 2.6)
project (Kinect-capture)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(Kinect-capture kinect_main.cpp mini-yaml/Yaml.cpp)
add_executable(Omnidir-capture omni_main.cpp mini-yaml/Yaml.cpp)

find_package(freenect2 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

include_directories("/usr/include/libusb-1.0/")

INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(Kinect-capture Threads::Threads)
target_link_libraries(Omnidir-capture Threads::Threads)
target_link_libraries(Kinect-capture ${freenect2_LIBRARIES})
target_link_libraries(Kinect-capture ${OpenCV_LIBRARIES})
target_link_libraries(Omnidir-capture ${OpenCV_LIBRARIES})

configure_file(launch.py launch.py COPYONLY)
configure_file(kinects.yaml kinects.yaml COPYONLY)