cmake_minimum_required(VERSION 3.22)

project(ceres_playground LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(rerun_sdk REQUIRED)

add_executable(hello_ceres
    src/01_hello_ceres.cpp
)

target_link_libraries(hello_ceres
    PRIVATE
        Ceres::ceres
)

add_executable(hello_rerun
    src/02_hello_rerun.cpp
)

target_link_libraries(hello_rerun
    PRIVATE
        rerun_sdk
)