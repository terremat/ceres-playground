#include <rerun.hpp>

#include <vector>

int main() {
    // Create a Rerun recording.
    const auto rec = rerun::RecordingStream("ceres_playground");

    // Spawn the Rerun Viewer and connect to it.
    rec.spawn().exit_on_failure();

    // Some simple 3D points.
    const std::vector<rerun::Position3D> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
    };

    // Log them under the entity path "world/points".
    rec.log(
        "world/points",
        rerun::Points3D(points)
    );

    return 0;
}