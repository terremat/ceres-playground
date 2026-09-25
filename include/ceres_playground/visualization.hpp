#pragma once

#include "ceres_playground/types.hpp"

#include <rerun.hpp>

#include <ceres/ceres.h>

#include <cstddef>
#include <string>
#include <vector>

inline void LogLandmarks(
    const rerun::RecordingStream& rec,
    const std::string& path,
    const std::vector<Eigen::Vector3d>& landmarks,
    bool is_static = false) {

    std::vector<rerun::Position3D> positions;
    positions.reserve(landmarks.size());

    for (const auto& p : landmarks) {
        positions.emplace_back(
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z()));
    }

    if (is_static) {
        rec.log_static(
            path,
            rerun::Points3D(positions));
    } else {
        rec.log(
            path,
            rerun::Points3D(positions));
    }
}

inline void LogCamera(
    const rerun::RecordingStream& rec,
    const std::string& path,
    const CameraPose& camera,
    const CameraIntrinsics& K,
    bool is_static = false) {

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

    const auto transform =
        rerun::Transform3D::from_translation_mat3x3({
            static_cast<float>(t_WC.x()),
            static_cast<float>(t_WC.y()),
            static_cast<float>(t_WC.z())
        }, rotation);

    const auto axes =
        rerun::TransformAxes3D(0.3);

    const auto pinhole =
        rerun::Pinhole::from_focal_length_and_resolution(
            {static_cast<float>(K.fx),
             static_cast<float>(K.fy)},
            {static_cast<float>(K.width),
             static_cast<float>(K.height)}
        );

    if (is_static) {
        rec.log_static(path, transform);
        rec.log_static(path, axes);
        rec.log_static(path, pinhole);
    } else {
        rec.log(path, transform);
        rec.log(path, axes);
        rec.log(path, pinhole);
    }
}

// Callback to visualize the camera poses after each iteration of the optimization.
class RerunPoseIterationCallback : public ceres::IterationCallback {
public:
    RerunPoseIterationCallback(
        const rerun::RecordingStream& rec,
        const std::vector<CameraParameters>& cameras,
        const CameraIntrinsics& K)
        : rec_(rec),
          cameras_(cameras),
          K_(K) {}

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary) override {

        rec_.set_time_sequence(
            "iteration",
            summary.iteration);

        for (std::size_t camera_id = 0;
             camera_id < cameras_.size();
             ++camera_id) {

            const std::string camera_name =
                "cam" + std::to_string(camera_id);

            LogCamera(
                rec_,
                "world/cameras/optimization/" + camera_name,
                ToCameraPose(cameras_[camera_id]),
                K_);
        }

        return ceres::SOLVER_CONTINUE;
    }

private:
    const rerun::RecordingStream& rec_;
    const std::vector<CameraParameters>& cameras_; // Reference to the camera parameters being optimized.
    CameraIntrinsics K_;
};