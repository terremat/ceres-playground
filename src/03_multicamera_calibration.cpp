#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

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


// -----------------------------------------------------------------------------
// Helper functions for generating synthetic data
// -----------------------------------------------------------------------------

std::vector<Eigen::Vector3d> GenerateLandmarks(
    int num_landmarks,
    std::mt19937& rng) {

    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> y_dist(-1.5, 1.5);
    std::uniform_real_distribution<double> z_dist(4.0, 8.0);

    std::vector<Eigen::Vector3d> landmarks;
    landmarks.reserve(num_landmarks);

    for (int i = 0; i < num_landmarks; ++i) {
        landmarks.emplace_back(
            x_dist(rng),
            y_dist(rng),
            z_dist(rng));
    }

    return landmarks;
}

bool ProjectPoint(
    const CameraIntrinsics& K,
    const Eigen::Vector3d& point_camera,
    Eigen::Vector2d* pixel) {

    // Point is behind the camera.
    if (point_camera.z() <= 0.0) {
        return false;
    }

    const double x = point_camera.x() / point_camera.z();
    const double y = point_camera.y() / point_camera.z();

    const double u = K.fx * x + K.cx;
    const double v = K.fy * y + K.cy;

    // Point falls outside the image.
    if (u < 0.0 || u >= K.width ||
        v < 0.0 || v >= K.height) {
        return false;
    }

    *pixel = Eigen::Vector2d(u, v);

    return true;
}

Eigen::Vector3d TransformWorldToCamera(
    const CameraPose& camera,
    const Eigen::Vector3d& point_world) {

    return camera.R_CW * point_world
         + camera.t_CW;
}

bool ProjectWorldPoint(
    const CameraIntrinsics& K,
    const CameraPose& camera,
    const Eigen::Vector3d& point_world,
    Eigen::Vector2d* pixel) {

    const Eigen::Vector3d point_camera =
        TransformWorldToCamera(
            camera,
            point_world);

    return ProjectPoint(
        K,
        point_camera,
        pixel);
}

int main() {
    std::mt19937 rng(42);

    // Generate 3D landmarks in the world frame
    const auto landmarks_W =
        GenerateLandmarks(100, rng);

    for (const auto& p : landmarks_W) {
        std::cout << p.transpose() << '\n';
    }

    CameraIntrinsics K;
    CameraPose camera;
    camera.R_CW = Eigen::Matrix3d::Identity();
    camera.t_CW = Eigen::Vector3d(-1, 0, 0);
    Eigen::Vector3d p_W(1.0, 0.0, 5.0);

    Eigen::Vector2d pixel;

    if (ProjectWorldPoint(
            K,
            camera,
            p_W,
            &pixel)) {

        std::cout
            << "P_W: "
            << p_W.transpose()
            << '\n';

        std::cout
            << "pixel: "
            << pixel.transpose()
            << '\n';
    }

    // Check the C++ standard being used
    std::cout << "C++ standard: " << __cplusplus << '(201703 = C++17)\n'; 

    return 0;
}