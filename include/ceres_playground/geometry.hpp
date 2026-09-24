#pragma once

#include "ceres_playground/types.hpp"

#include <ceres/rotation.h>

inline bool ProjectPoint(
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

inline Eigen::Vector3d TransformWorldToCamera(
    const CameraPose& camera,
    const Eigen::Vector3d& point_world) {

    return camera.R_CW * point_world
         + camera.t_CW;
}

inline bool ProjectWorldPoint(
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

inline CameraPose MakeCameraPose(
    const Eigen::Vector3d& position_world,
    const Eigen::Matrix3d& R_CW) {

    CameraPose camera;

    camera.R_CW = R_CW;
    camera.t_CW = -R_CW * position_world;

    return camera;
}

inline CameraParameters ToParameters(
    const CameraPose& pose) {

    CameraParameters parameters;

    ceres::RotationMatrixToAngleAxis(
        pose.R_CW.data(),
        parameters.values.data());

    parameters.values[3] = pose.t_CW.x();
    parameters.values[4] = pose.t_CW.y();
    parameters.values[5] = pose.t_CW.z();

    return parameters;
}

inline CameraPose ToCameraPose(
    const CameraParameters& parameters) {

    CameraPose pose;

    ceres::AngleAxisToRotationMatrix(
        parameters.values.data(),
        pose.R_CW.data());

    pose.t_CW = Eigen::Vector3d(
        parameters.values[3],
        parameters.values[4],
        parameters.values[5]);

    return pose;
}
