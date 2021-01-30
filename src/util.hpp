#pragma once

#include <functional>
#include <Eigen/Dense>

namespace scf {

struct vec3 {
    double x, y, z;
};

constexpr double PI = 3.141592653589793238462643;

using BasisFunction = std::function<double(const vec3&)>;

double mc_integrator(std::function<double(const std::vector<double>&)> fn, const unsigned long n,
    const std::size_t dim, const std::vector<double> bounds);

double mc_normal_int(std::function<double(const std::vector<double>&)> fn, const unsigned long n,
    const std::size_t dim, Eigen::VectorXd var, Eigen::VectorXd mean);

double gto(const vec3& coord, const vec3& center, const double c, const double alpha);

double gto_gradient(const vec3& coord, const vec3& center, const double c, const double alpha);

}
