#pragma once

#include <Eigen/Core>

#include <array>

struct CameraIntrinsics {
    double fx = 500.0;
    double fy = 500.0;
    double cx = 320.0;
    double cy = 240.0;
    int width = 640;
    int height = 480;
};

struct CameraPose {
    Eigen::Matrix3d R_CW = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_CW = Eigen::Vector3d::Zero();
};

struct Observation {
    int camera_id;
    int landmark_id;
    Eigen::Vector2d pixel;
};

struct CameraParameters {
    std::array<double, 6> values;
};
