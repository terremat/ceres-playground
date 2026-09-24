#pragma once

#include "ceres_playground/geometry.hpp"
#include "ceres_playground/types.hpp"

#include <cstddef>
#include <random>
#include <vector>

inline std::vector<Eigen::Vector3d> GenerateLandmarks(
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

inline std::vector<CameraPose> GenerateCameras() {

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

inline std::vector<Observation> GenerateObservations(
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

inline std::vector<CameraParameters> PerturbCameras(
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



inline void AddGaussianNoise(
    std::vector<Observation>& observations,
    double sigma_pixels,
    std::mt19937& rng) {

    std::normal_distribution<double> noise(
        0.0,
        sigma_pixels);

    for (auto& observation : observations) {
        observation.pixel.x() += noise(rng);
        observation.pixel.y() += noise(rng);
    }
}