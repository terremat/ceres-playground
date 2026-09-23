#pragma once

#include "ceres_playground/types.hpp"

#include <rerun.hpp>

#include <string>
#include <vector>

inline void LogLandmarks(
    const rerun::RecordingStream& rec,
    const std::vector<Eigen::Vector3d>& landmarks) {

    std::vector<rerun::Position3D> positions;
    positions.reserve(landmarks.size());

    for (const auto& p : landmarks) {
        positions.emplace_back(
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z()));
    }

    rec.log(
        "world/landmarks",
        rerun::Points3D(positions));
}

inline void LogCamera(
    const rerun::RecordingStream& rec,
    const std::string& path,
    const CameraPose& camera,
    const CameraIntrinsics& K) {

    const Eigen::Matrix3d R_WC =
        camera.R_CW.transpose();

    const Eigen::Vector3d t_WC =
        -R_WC * camera.t_CW;

    // Rerun expects the camera-to-world transform, including orientation.
    const rerun::components::TransformMat3x3 rotation({
        static_cast<float>(R_WC(0, 0)),
        static_cast<float>(R_WC(1, 0)),
        static_cast<float>(R_WC(2, 0)),
        static_cast<float>(R_WC(0, 1)),
        static_cast<float>(R_WC(1, 1)),
        static_cast<float>(R_WC(2, 1)),
        static_cast<float>(R_WC(0, 2)),
        static_cast<float>(R_WC(1, 2)),
        static_cast<float>(R_WC(2, 2))
    });

    rec.log(
        path,
        rerun::Transform3D::from_translation_mat3x3({
            static_cast<float>(t_WC.x()),
            static_cast<float>(t_WC.y()),
            static_cast<float>(t_WC.z())
        }, rotation));

    rec.log(
        path,
        rerun::TransformAxes3D(0.3));

    rec.log(
        path,
        rerun::Pinhole::from_focal_length_and_resolution(
            {static_cast<float>(K.fx),
             static_cast<float>(K.fy)},
            {static_cast<float>(K.width),
             static_cast<float>(K.height)}
        )
    );
}
