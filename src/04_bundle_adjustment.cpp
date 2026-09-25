#include "ceres_playground/types.hpp"
#include "ceres_playground/geometry.hpp"
#include "ceres_playground/metrics.hpp"
#include "ceres_playground/synthetic.hpp"
#include "ceres_playground/visualization.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

#include <cstddef>
#include <iostream>
#include <random>
#include <string>

// Pose-only estimation: intrinsics, 3D landmarks, and observations are fixed.
// Each camera has six variable parameters: angle-axis rotation and translation.
// Unlike lesson 01's scalar residual, each observation produces two pixel residuals.
// This is not bundle adjustment or full multicamera calibration.

// Ceres cost functor for reprojection error of a single landmark observation.
struct ReprojectionError {

    ReprojectionError(
        const Eigen::Vector2d& observed_pixel,
        const CameraIntrinsics& K)
        : observed_pixel_(observed_pixel),
          K_(K) {}

    template <typename T>
    bool operator()(
        const T* const camera,
        const T* const landmark,
        T* residuals) const {

        // Rotate P_W into the camera frame.
        T point_camera[3];

        ceres::AngleAxisRotatePoint(
            camera,
            landmark,
            point_camera);

        // Translation t_CW.
        point_camera[0] += camera[3];
        point_camera[1] += camera[4];
        point_camera[2] += camera[5];

        // Perspective division.
        const T x =
            point_camera[0] / point_camera[2];

        const T y =
            point_camera[1] / point_camera[2];

        // Pinhole projection.
        const T predicted_u =
            T(K_.fx) * x + T(K_.cx);

        const T predicted_v =
            T(K_.fy) * y + T(K_.cy);

        // Reprojection error.
        residuals[0] =
            predicted_u - T(observed_pixel_.x());

        residuals[1] =
            predicted_v - T(observed_pixel_.y());

        return true;
    }

private:
    Eigen::Vector2d observed_pixel_;
    CameraIntrinsics K_;
};

int main() {

    // 1. Generate synthetic ground truth.
    constexpr int kNumLandmarks = 100;
    std::mt19937 rng(42);

    // Generate 3D landmarks in the world frame
    const auto landmarks_gt =
        GenerateLandmarks(kNumLandmarks, rng);

    // Ground-truth camera rig.
    const CameraIntrinsics K;
    const auto cameras_gt =
        GenerateCameras();

    // 2. Generate fixed observations from the ground truth.
    const auto observations =
        GenerateObservations(
            K,
            cameras_gt,
            landmarks_gt);

    // 3. Create perturbed initial camera estimates.
    auto cameras_estimated =
        PerturbCameras(cameras_gt, rng);

    const auto cameras_initial = cameras_estimated;  // snapshot before optimization

    const double initial_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_gt,
            observations);

    auto landmarks_estimated = landmarks_gt;
    
    const double initial_landmark_rmse =
        ComputeLandmarkRMSE(
            landmarks_estimated,
            landmarks_gt);

    // 4. Build the problem. Only the six camera parameters are variable.
    ceres::Problem problem;

    for (const auto& obs : observations) {
        const auto& pixel = obs.pixel;

        // Two residuals (u, v), one camera parameter block of size six and one landmark parameter block of size three.
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(
                    pixel,
                    K
                )
            );

        problem.AddResidualBlock(
            cost_function,
            nullptr,
            cameras_estimated[obs.camera_id].values.data(),     // 6
            landmarks_estimated[obs.landmark_id].data()         // 3
        );
    }

    // Gauge fix: use camera 0 as the reference frame (anchor) by keeping its parameters constant (no updates during optimization).
    problem.SetParameterBlockConstant(
        cameras_estimated[0].values.data());

    // 5. Configure the solver, just as in lesson 01.
    ceres::Solver::Options options;

    options.linear_solver_type =
        ceres::DENSE_QR;

    options.minimizer_progress_to_stdout =
        true;

    // 6. Solve. Ceres updates the camera parameters in place.
    ceres::Solver::Summary summary;

    ceres::Solve(
        options,
        &problem,
        &summary);

    // 7. Evaluate and visualize.
    std::cout
        << summary.BriefReport()
        << '\n';

    const double final_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_estimated,
            observations);

    std::cout
        << "Reprojection RMSE: "
        << initial_rmse
        << " px -> "
        << final_rmse
        << " px\n";


    const double final_landmark_rmse =
        ComputeLandmarkRMSE(
            landmarks_estimated,
            landmarks_gt);

    std::cout
        << "Landmark RMSE: "
        << initial_landmark_rmse
        << " m -> "
        << final_landmark_rmse
        << " m\n";
        

    // Compare ground-truth, initial, and optimized poses in separate Rerun groups.
    const auto rec =
        rerun::RecordingStream(
            "pose_estimation");

    rec.spawn().exit_on_failure();

    LogLandmarks(rec, landmarks_estimated);

    for (std::size_t camera_id = 0;
         camera_id < cameras_gt.size();
         ++camera_id) {

        const std::string camera_name = "cam" + std::to_string(camera_id);

        LogCamera(rec, "world/cameras/gt/" + camera_name,
                  cameras_gt[camera_id], K);
        LogCamera(rec, "world/cameras/initial/" + camera_name,
                  ToCameraPose(cameras_initial[camera_id]), K);
        LogCamera(rec, "world/cameras/optimized/" + camera_name,
                  ToCameraPose(cameras_estimated[camera_id]), K);
    }

    return 0;
}
