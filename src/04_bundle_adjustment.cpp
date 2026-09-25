// Bundle adjustment extends pose estimation by optimizing 3D landmarks too.
// Each 2D reprojection residual connects a 6-parameter camera pose (angle-axis
// and translation) to a 3-parameter landmark. Intrinsics and observations stay fixed.
// Two ground-truth camera poses anchor the world frame and metric scale;
// the remaining camera pose and the landmarks are recovered from perturbed estimates.

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

        // Rotate the variable landmark from world coordinates into the camera frame.
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
    constexpr double kLandmarkNoiseMeters = 0.1;
    std::mt19937 rng(42);

    // Generate ground-truth 3D landmarks in the world frame.
    const auto landmarks_gt =
        GenerateLandmarks(kNumLandmarks, rng);

    // Ground-truth camera rig.
    const CameraIntrinsics K;
    const auto cameras_gt =
        GenerateCameras();

    // 2. Generate fixed observations from ground-truth cameras and landmarks.
    const auto observations =
        GenerateObservations(
            K,
            cameras_gt,
            landmarks_gt);

    // 3. Perturb both camera poses and landmarks to create initial estimates.
    auto cameras_estimated =
        PerturbCameras(cameras_gt, rng);

    auto landmarks_estimated =
        PerturbLandmarks(
            landmarks_gt,
            kLandmarkNoiseMeters,
            rng);

    // Restore the first two poses to ground truth before fixing them below.
    cameras_estimated[0] =
        ToParameters(cameras_gt[0]);

    cameras_estimated[1] =
        ToParameters(cameras_gt[1]);

    const auto cameras_initial = cameras_estimated;  // snapshot before optimization

    // Evaluate the initial reprojection error using the perturbed state.
    const double initial_reprojection_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_estimated,
            observations);

    const auto landmarks_initial = landmarks_estimated;
    
    const double initial_landmark_rmse =
        ComputeLandmarkRMSE(
            landmarks_estimated,
            landmarks_gt);

    // 4. Build the problem with camera and landmark parameter blocks.
    ceres::Problem problem;

    for (const auto& obs : observations) {
        const auto& pixel = obs.pixel;

        // Two residuals (u, v) depend on two parameter blocks:
        // six camera parameters and three landmark coordinates.
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

    // Fix camera 0 at its ground-truth pose to anchor the world frame.
    problem.SetParameterBlockConstant(
        cameras_estimated[0].values.data());
    
    // Fix camera 1 at ground truth too: the known baseline sets metric scale.
    // Together these anchors remove gauge freedom.
    problem.SetParameterBlockConstant(
        cameras_estimated[1].values.data());

    // 5. Keep DENSE_QR for this lesson; Schur solvers come later.
    ceres::Solver::Options options;

    options.linear_solver_type =
        ceres::DENSE_QR;

    options.minimizer_progress_to_stdout =
        true;

    // 6. Solve. Ceres updates the remaining camera pose and landmarks in place.
    ceres::Solver::Summary summary;

    ceres::Solve(
        options,
        &problem,
        &summary);

    // 7. Evaluate and visualize.
    std::cout
        << summary.BriefReport()
        << '\n';

    const double final_reprojection_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_estimated,
            observations);

    std::cout
        << "Reprojection RMSE: "
        << initial_reprojection_rmse
        << " px -> "
        << final_reprojection_rmse
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
        

    // Compare ground-truth, initial, and optimized cameras and landmarks in Rerun.
    const auto rec =
        rerun::RecordingStream(
            "bundle_adjustment");

    rec.spawn().exit_on_failure();

    LogLandmarks(
        rec,
        "world/landmarks/gt",
        landmarks_gt,
        true);

    LogLandmarks(
        rec,
        "world/landmarks/initial",
        landmarks_initial,
        true);

    LogLandmarks(
        rec,
        "world/landmarks/optimized",
        landmarks_estimated,
        true);

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
