#pragma once

#include "ceres_playground/geometry.hpp"
#include "ceres_playground/types.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

inline double ComputeReprojectionRMSE(
    const CameraIntrinsics& K,
    const std::vector<CameraParameters>& cameras,
    const std::vector<Eigen::Vector3d>& landmarks,
    const std::vector<Observation>& observations) {

    double squared_error_sum = 0.0;

    for (const auto& observation : observations) {

        const CameraPose camera =
            ToCameraPose(
                cameras[observation.camera_id]);

        Eigen::Vector2d predicted_pixel;

        const bool visible =
            ProjectWorldPoint(
                K,
                camera,
                landmarks[observation.landmark_id],
                &predicted_pixel);

        if (!visible) {
            continue;
        }

        const Eigen::Vector2d error =
            predicted_pixel - observation.pixel;

        squared_error_sum += error.squaredNorm();
    }

    return std::sqrt(
        squared_error_sum /
        static_cast<double>(observations.size()));
}



inline double ComputeLandmarkRMSE(
    const std::vector<Eigen::Vector3d>& landmarks_estimated,
    const std::vector<Eigen::Vector3d>& landmarks_gt) {

    double squared_error_sum = 0.0;

    for (std::size_t i = 0;
         i < landmarks_estimated.size();
         ++i) {

        const Eigen::Vector3d error =
            landmarks_estimated[i] - landmarks_gt[i];

        squared_error_sum += error.squaredNorm();
    }

    return std::sqrt(
        squared_error_sum /
        static_cast<double>(landmarks_estimated.size()));
}