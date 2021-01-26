#pragma once

#include <Eigen/Dense>
#include <bits/c++config.h>
#include <functional>
#include <utility>

namespace scf {

struct vec3 {
    double x, y, z;
};

using BasisFunction = std::function<double(const vec3&)>;

double mc_integrator(std::function<double(const std::vector<double>&)> fn, const unsigned long n,
    const std::size_t dim, const std::vector<double> bounds);

double gto(const vec3& coord, const vec3& center, const double c, const double alpha);

double gto_gradient(const vec3& coord, const vec3& center, const double c, const double alpha);

class HartreeFockSolver {
private:
    const std::vector<BasisFunction> m_basisFunctions;
    const std::vector<BasisFunction> m_basisGradients;
    const std::vector<std::pair<vec3, double>> m_nuclei;

    const unsigned long m_sampleSize = 100000000000;
    const double m_lowerBound = -5.0;
    const double m_upperBound = 5.0;

    const std::size_t m_size;

    Eigen::MatrixXd m_kinEnergy;
    std::vector<Eigen::MatrixXd> m_potEnergies; // For each nucleus.
    Eigen::MatrixXd m_electronRepulsion;
    std::vector<std::vector<std::vector<std::vector<double>>>> m_repulsionIntegrals;

    void calcKineticEnergy();
    void calcPotential();
    void calcRepulsionIntegrals();

public:
    HartreeFockSolver(std::vector<BasisFunction> basis_functions,
        std::vector<BasisFunction> basis_gradients,
        std::vector<std::pair<vec3, double>> nuclei);

    double solveSCF();
};

};
