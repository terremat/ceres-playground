#include "ceres_playground/types.hpp"
#include "ceres_playground/geometry.hpp"
#include "ceres_playground/metrics.hpp"
#include "ceres_playground/synthetic.hpp"
#include "ceres_playground/visualization.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

// This lesson introduces noisy pixel measurements, whitened residuals, and pose covariance.
// Reprojection residuals are divided by the assumed pixel noise sigma so that
// Ceres covariance reflects the measurement uncertainty.
// Only camera poses are optimized; intrinsics and 3D landmarks remain fixed.
// Camera 0's 6x6 covariance describes its angle-axis and translation parameters.

// Ceres cost functor for reprojection error of a single landmark observation.
struct ReprojectionError {

    ReprojectionError(
        const Eigen::Vector3d& point_world,
        const Eigen::Vector2d& observed_pixel,
        const CameraIntrinsics& K,
        double sigma_pixels)
        : point_world_(point_world),
          observed_pixel_(observed_pixel),
          K_(K),
          sigma_pixels_(sigma_pixels) {}

    template <typename T>
    bool operator()(
        const T* const camera,
        T* residuals) const {

        // 3D landmark expressed in world coordinates.
        const T point_world[3] = {
            T(point_world_.x()),
            T(point_world_.y()),
            T(point_world_.z())
        };

        // Rotate P_W into the camera frame.
        T point_camera[3];

        ceres::AngleAxisRotatePoint(
            camera,
            point_world,
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
            (predicted_u - T(observed_pixel_.x()))
            / T(sigma_pixels_);

        residuals[1] =
            (predicted_v - T(observed_pixel_.y()))
            / T(sigma_pixels_);

        return true;
    }

private:
    Eigen::Vector3d point_world_;
    Eigen::Vector2d observed_pixel_;
    CameraIntrinsics K_;
    double sigma_pixels_;
};

int main() {

    // 1. Generate synthetic ground truth.
    constexpr int kNumLandmarks = 100;
    constexpr double kNoiseSigmaPixels = 1.0;
    std::mt19937 rng(42);

    // Generate 3D landmarks in the world frame
    const auto landmarks_gt =
        GenerateLandmarks(kNumLandmarks, rng);

    // Ground-truth camera rig.
    const CameraIntrinsics K;
    const auto cameras_gt =
        GenerateCameras();

    // 2. Generate noisy observations from the ground truth.
    const auto observations_gt =
        GenerateObservations(
            K,
            cameras_gt,
            landmarks_gt);

    auto observations_noisy =
        observations_gt;

    AddGaussianNoise(
        observations_noisy,
        kNoiseSigmaPixels,
        rng);

    // 3. Create perturbed initial camera estimates.
    auto cameras_estimated =
        PerturbCameras(cameras_gt, rng);

    const auto cameras_initial = cameras_estimated;  // snapshot before optimization

    const double initial_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_gt,
            observations_noisy);

    // 4. Build the problem. Only the six camera parameters are variable.
    ceres::Problem problem;

    for (const auto& obs : observations_noisy) {
        const auto& point = landmarks_gt[obs.landmark_id];
        const auto& pixel = obs.pixel;

        // Two residuals (u, v), one parameter block of size six.
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
                new ReprojectionError(
                    point,
                    pixel,
                    K,
                    kNoiseSigmaPixels)
            );

        problem.AddResidualBlock(
            cost_function,
            nullptr,
            cameras_estimated[obs.camera_id].values.data()
        );
    }

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

    // Covariance computation, useful for uncertainty analysis.
    ceres::Covariance::Options covariance_options;
    ceres::Covariance covariance(covariance_options);
    
    const double* camera_parameters =
        cameras_estimated[0].values.data();

    // asking Cov(camera0, camera0) 6x6 matrix
    std::vector<std::pair<const double*, const double*>>
        covariance_blocks = {
            {camera_parameters, camera_parameters}
        };

    if (!covariance.Compute(
            covariance_blocks,
            &problem)) {

        std::cerr
            << "Failed to compute covariance.\n";

        return 1;
    }

    // 7. Evaluate and visualize.
    std::cout
        << summary.BriefReport()
        << '\n';

    const double noisy_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_gt,
            observations_noisy);

    const double gt_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_gt,
            observations_gt);

    std::cout
        << "Reprojection RMSE: "
        << initial_rmse
        << " px -> "
        << noisy_rmse
        << " px | "
        << gt_rmse
        << " px (GT)\n";

    double covariance_matrix[6 * 6];

    covariance.GetCovarianceBlock(
        camera_parameters,
        camera_parameters,
        covariance_matrix);

    std::cout
        << "\nAssumed pixel noise sigma: "
        << kNoiseSigmaPixels
        << " px\n";

    std::cout
        << "\nCamera 0 covariance:\n";

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            std::cout
                << covariance_matrix[row * 6 + col]
                << '\t';
        }

        std::cout << '\n';
    }
    std::cout
        << "\nCamera 0 standard deviations:\n";

    std::cout
        << "rotation [rad]: "
        << std::sqrt(covariance_matrix[0 * 6 + 0]) << ' '
        << std::sqrt(covariance_matrix[1 * 6 + 1]) << ' '
        << std::sqrt(covariance_matrix[2 * 6 + 2]) << '\n';

    std::cout
        << "translation [m]: "
        << std::sqrt(covariance_matrix[3 * 6 + 3]) << ' '
        << std::sqrt(covariance_matrix[4 * 6 + 4]) << ' '
        << std::sqrt(covariance_matrix[5 * 6 + 5]) << '\n';


    // Compare ground-truth, initial, and optimized poses in separate Rerun groups.
    const auto rec =
        rerun::RecordingStream(
            "covariance");

    rec.spawn().exit_on_failure();

    LogLandmarks(rec, "world/landmarks/gt", landmarks_gt, true);

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
