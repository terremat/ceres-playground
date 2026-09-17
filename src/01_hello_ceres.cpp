#include <ceres/ceres.h>

#include <iostream>

struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

int main() {
    double x = 5.0;

    const double initial_x = x;

    ceres::Problem problem;

    auto* cost_function =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(
            new CostFunctor);

    problem.AddResidualBlock(
        cost_function,
        nullptr,
        &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << '\n';
    std::cout << "x: " << initial_x << " -> " << x << '\n';

    return 0;
}