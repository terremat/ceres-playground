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

#include <rerun.hpp>

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

CameraPose MakeCameraPose(
    const Eigen::Vector3d& position_world,
    const Eigen::Matrix3d& R_CW) {

    CameraPose camera;

    camera.R_CW = R_CW;
    camera.t_CW = -R_CW * position_world;

    return camera;
}

std::vector<CameraPose> GenerateCameras() {

    const Eigen::Matrix3d R_CW =
        Eigen::Matrix3d::Identity();

    std::vector<CameraPose> cameras;

    cameras.push_back(
        MakeCameraPose(
            Eigen::Vector3d(-1.0, 0.0, 0.0),
            R_CW));

    cameras.push_back(
        MakeCameraPose(
            Eigen::Vector3d(0.0, 0.0, 0.0),
            R_CW));

    cameras.push_back(
        MakeCameraPose(
            Eigen::Vector3d(1.0, 0.0, 0.0),
            R_CW));

    return cameras;
}

std::vector<Observation> GenerateObservations(
    const CameraIntrinsics& K,
    const std::vector<CameraPose>& cameras,
    const std::vector<Eigen::Vector3d>& landmarks) {

    std::vector<Observation> observations;

    for (std::size_t camera_id = 0;
         camera_id < cameras.size();
         ++camera_id) {

        for (std::size_t landmark_id = 0;
             landmark_id < landmarks.size();
             ++landmark_id) {

            Eigen::Vector2d pixel;

            const bool visible =
                ProjectWorldPoint(
                    K,
                    cameras[camera_id],
                    landmarks[landmark_id],
                    &pixel);

            if (!visible) {
                continue;
            }

            observations.push_back({
                static_cast<int>(camera_id),
                static_cast<int>(landmark_id),
                pixel
            });
        }
    }

    return observations;
}

// -----------------------------------------------------------------------------
// Helper functions for converting between Ceres and our data structures
// -----------------------------------------------------------------------------
struct CameraParameters {
    std::array<double, 6> values;
};

CameraParameters ToParameters(
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

CameraPose ToCameraPose(
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

std::vector<CameraParameters> PerturbCameras(
    const std::vector<CameraPose>& cameras_gt,
    std::mt19937& rng) {

    std::normal_distribution<double> rotation_noise(
        0.0, 0.02);

    std::normal_distribution<double> translation_noise(
        0.0, 0.05);

    std::vector<CameraParameters> cameras_initial;
    cameras_initial.reserve(cameras_gt.size());

    for (const auto& camera_gt : cameras_gt) {

        CameraParameters parameters =
            ToParameters(camera_gt);

        // Perturb angle-axis rotation.
        parameters.values[0] += rotation_noise(rng);
        parameters.values[1] += rotation_noise(rng);
        parameters.values[2] += rotation_noise(rng);

        // Perturb translation t_CW.
        parameters.values[3] += translation_noise(rng);
        parameters.values[4] += translation_noise(rng);
        parameters.values[5] += translation_noise(rng);

        cameras_initial.emplace_back(parameters);
    }

    return cameras_initial;
}

// Ceres cost functor for reprojection error of a single landmark observation.
struct ReprojectionError {

    ReprojectionError(
        const Eigen::Vector3d& point_world,
        const Eigen::Vector2d& observed_pixel,
        const CameraIntrinsics& K)
        : point_world_(point_world),
          observed_pixel_(observed_pixel),
          K_(K) {}

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
            predicted_u - T(observed_pixel_.x());

        residuals[1] =
            predicted_v - T(observed_pixel_.y());

        return true;
    }

private:
    Eigen::Vector3d point_world_;
    Eigen::Vector2d observed_pixel_;
    CameraIntrinsics K_;
};

double ComputeReprojectionRMSE(
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


// -----------------------------------------------------------------------------
// Logging functions
// -----------------------------------------------------------------------------
void LogLandmarks(
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

void LogCamera(
    const rerun::RecordingStream& rec,
    const std::string& path,
    const CameraPose& camera,
    const CameraIntrinsics& K) {

    const Eigen::Matrix3d R_WC =
        camera.R_CW.transpose();

    const Eigen::Vector3d t_WC =
        -R_WC * camera.t_CW;

    rec.log(
        path,
        rerun::Transform3D::from_translation({
            static_cast<float>(t_WC.x()),
            static_cast<float>(t_WC.y()),
            static_cast<float>(t_WC.z())
        }));

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


int main() {

    constexpr int kNumLandmarks = 100;

    std::mt19937 rng(42);

    // Generate 3D landmarks in the world frame
    const auto landmarks_W =
        GenerateLandmarks(kNumLandmarks, rng);

    for (const auto& p : landmarks_W) {
        std::cout << p.transpose() << '\n';
    }

    // Ground-truth camera rig.
    const CameraIntrinsics K;
    const auto cameras =
        GenerateCameras();

    
    // Perfect synthetic image measurements.
    const auto observations =
        GenerateObservations(
            K,
            cameras,
            landmarks_W);

    std::cout
        << "Landmarks: "
        << landmarks_W.size()
        << '\n';

    std::cout
        << "Cameras: "
        << cameras.size()
        << '\n';

    std::cout
        << "Observations: "
        << observations.size()
        << '\n';


    for (std::size_t camera_id = 0;
     camera_id < cameras.size();
     ++camera_id) {

        int count = 0;

        for (const auto& obs : observations) {
            if (obs.camera_id ==
                static_cast<int>(camera_id)) {
                ++count;
            }
        }

        std::cout
            << "Camera "
            << camera_id
            << " sees "
            << count
            << " landmarks\n";
    }


    // ---- Rerun ----
    const auto rec =
        rerun::RecordingStream(
            "multicamera_calibration");

    rec.spawn().exit_on_failure();

    LogLandmarks(rec, landmarks_W);

    for (std::size_t camera_id = 0;
        camera_id < cameras.size();
        ++camera_id) {

        const std::string path =
            "world/cameras/cam" +
            std::to_string(camera_id);

        LogCamera(
            rec,
            path,
            cameras[camera_id],
            K); 
    }





    // CERES SOLVER SETUP
    for (std::size_t i = 0; i < cameras.size(); ++i) {

        const auto parameters =
            ToParameters(cameras[i]);

        std::cout
            << "Camera " << i << '\n'
            << "rotation: "
            << parameters.values[0] << " "
            << parameters.values[1] << " "
            << parameters.values[2] << '\n'
            << "translation: "
            << parameters.values[3] << " "
            << parameters.values[4] << " "
            << parameters.values[5] << '\n';
    }

    auto cameras_estimated =
    PerturbCameras(
        cameras,
        rng);

        
    const auto cameras_initial = cameras_estimated;  // snapshot before optimization

    // Create a Ceres problem
    ceres::Problem problem;

    for (const auto& obs : observations) {
        const auto& point = landmarks_W[obs.landmark_id];
        const auto& pixel = obs.pixel;
        const auto& camera = cameras_estimated[obs.camera_id];

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
                new ReprojectionError(
                    point,
                    pixel,
                    K
                )
            );

        problem.AddResidualBlock(
            cost_function,  
            nullptr,
            cameras_estimated[obs.camera_id].values.data()
        );
    }

    const double initial_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_W,
            observations);

    ceres::Solver::Options options;

    options.linear_solver_type =
        ceres::DENSE_QR;

    options.minimizer_progress_to_stdout =
        true;

    ceres::Solver::Summary summary;

    ceres::Solve(
        options,
        &problem,
        &summary);

    std::cout
        << summary.BriefReport()
        << '\n';

    for (std::size_t i = 0; i < cameras.size(); ++i) {

        const auto gt = ToParameters(cameras[i]);

        std::cout << "\nCamera " << i << '\n';

        std::cout << "GT:        ";
        for (double x : gt.values)
            std::cout << x << " ";

        std::cout << "\nInitial:   ";
        for (double x : cameras_initial[i].values)
            std::cout << x << " ";

        std::cout << "\nOptimized: ";
        for (double x : cameras_estimated[i].values)
            std::cout << x << " ";

        std::cout << '\n';
    }

    const double final_rmse =
        ComputeReprojectionRMSE(
            K,
            cameras_estimated,
            landmarks_W,
            observations);

    std::cout
        << "Reprojection RMSE: "
        << initial_rmse
        << " px -> "
        << final_rmse
        << " px\n";


    return 0;

    
}