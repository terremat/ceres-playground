cmake_minimum_required(VERSION 3.22)

project(ceres_playground LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
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
    examples/hello_rerun.cpp
)

target_link_libraries(hello_rerun
    PRIVATE
        rerun_sdk
)

add_executable(pose_estimation
    src/02_pose_estimation.cpp
)

target_include_directories(pose_estimation
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pose_estimation
    PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        rerun_sdk
)

add_executable(robust_estimation
    src/03_robust_estimation.cpp
)

target_include_directories(robust_estimation
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(robust_estimation
    PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        rerun_sdk
)