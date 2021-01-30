#include "util.hpp"

#include <Eigen/Dense>

#include <assert.h>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

double scf::mc_integrator(std::function<double(const std::vector<double>&)> fn,
    const unsigned long n, const std::size_t dim, const std::vector<double> bounds)
{
    assert(dim * 2 == bounds.size());

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::vector<std::uniform_real_distribution<double>> distributions;
    distributions.reserve(dim);
    for (unsigned int i = 0; i < dim; i++)
        distributions.push_back(
            std::uniform_real_distribution<double>(bounds[i * 2], bounds[i * 2 + 1]));

    double sum = 0;

    std::vector<double> values(dim);

    for (unsigned long i = 0; i < n; i++) {
        // Generate values for each dimension within bounds.
        for (unsigned int j = 0; j < dim; j++)
            values[j] = distributions[j](generator);

        sum += fn(values);
    }

    double volume = 1;
    for (unsigned int i = 0; i < dim * 2; i += 2)
        volume *= bounds[i + 1] - bounds[i];

    sum *= volume / n;

    return sum;
}

double scf::mc_normal_int(std::function<double(const std::vector<double>&)> fn,
    const unsigned long n, const std::size_t dim, Eigen::VectorXd var, Eigen::VectorXd mean)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::vector<std::normal_distribution<double>> distributions;
    distributions.reserve(dim);
    for (unsigned int i = 0; i < dim; i++)
        distributions.push_back(std::normal_distribution<double>(mean(i), std::sqrt(var(i))));

    auto pdf = [=](const std::vector<double>& pos) -> double {
        Eigen::VectorXd x(pos.size());

        for (unsigned int i = 0; i < pos.size(); i++)
            x(i) = pos[i];

        return (1.0 / std::sqrt(std::pow(2*PI, dim) * var.prod()))
            * std::exp(
                -0.5 * ((x - mean).transpose() * var.asDiagonal().inverse() * (x - mean))(0));
    };

    double sum = 0;
    std::vector<double> values(dim);

    for (unsigned long i = 0; i < n; i++) {
        for (unsigned int j = 0; j < dim; j++) {
            values[j] = distributions[j](generator);
            // std::cout << values[j] << ' ';
        }

        // std::cout << pdf(values)<< "\n\n";
        sum += fn(values) / pdf(values);
    }

    return sum / n;
}

double scf::gto(const vec3& coord, const vec3& center, const double c, const double alpha)
{
    double distance_square = (coord.x - center.x) * (coord.x - center.x)
        + (coord.y - center.y) * (coord.y - center.y) + (coord.z - center.z) * (coord.z - center.z);
    return c * std::exp(-alpha * distance_square);
}

double scf::gto_gradient(const vec3& coord, const vec3& center, const double c, const double alpha)
{
    double x = coord.x, y = coord.y, z = coord.z, x0 = center.x, y0 = center.y, z0 = center.z;

    double distance_square = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
    double xterm = 2 * alpha * x * x - 4 * x0 * alpha * x + 2 * x0 * x0 * alpha;
    double yterm = 2 * alpha * y * y - 4 * y0 * alpha * y + 2 * y0 * y0 * alpha;
    double zterm = 2 * alpha * z * z - 4 * z0 * alpha * z + 2 * z0 * z0 * alpha;
    return 2 * alpha * c * (xterm + yterm + zterm - 3) * std::exp(-alpha * distance_square);
}
