set(EXEC_NAME online-stereo-learning)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(EXECUTABLE_OUTPUT_PATH ".")

set(CUDA_VERSION "7.5")
set(OCV_VERSION "3.1")
        
find_package(ZED 1.0 REQUIRED)
find_package(Eigen3 REQUIRED)

##For Jetson, OpenCV4Tegra is based on OpenCV2.4
exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
    set(OCV_VERSION "2.4")
    set(CUDA_VERSION "")
else()
    set(OCV_MODULE "imgcodecs")
endif()

find_package(OpenCV ${OCV_VERSION} COMPONENTS core highgui imgproc calib3d ${OCV_MODULE} REQUIRED)
find_package(CUDA ${CUDA_VERSION} REQUIRED)

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(SRC_FOLDER src)
file(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")
 
add_executable(${EXEC_NAME} ${SRC_FILES})
set_property(TARGET ${EXEC_NAME} PROPERTY OUTPUT_NAME ${EXEC_NAME})

add_definitions(-std=c++11)# -m64)

target_link_libraries(${EXEC_NAME} ${ZED_LIBRARIES} ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" ) # for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra" ) # for debugging

option(WITH_ZED_SDK "Use the ZED SDK for depth estimation" ON)

if(WITH_ZED_SDK)
    add_definitions(-DUSE_ZED_SDK=1)
else()
    add_definitions(-DUSE_ZED_SDK=0)
endif()
