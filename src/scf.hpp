#pragma once
#include "util.hpp"

#include <Eigen/Dense>

#include <bits/c++config.h>
#include <functional>
#include <utility>

namespace scf {

class HartreeFockSolver {
private:
    const std::vector<BasisFunction> m_basisFunctions;
    const std::vector<BasisFunction> m_basisGradients;
    const std::vector<std::pair<vec3, double>> m_nuclei;

    const unsigned long m_sampleSize = 200000000;
    const double m_lowerBound = -4.0;
    const double m_upperBound = 4.0;

    const std::size_t m_size;
    const unsigned int m_electrons;

    Eigen::MatrixXd m_density;
    Eigen::MatrixXd m_overlap;

    Eigen::MatrixXd m_kinEnergy;
    std::vector<Eigen::MatrixXd> m_potEnergies; // For each nucleus.
    Eigen::MatrixXd m_electronRepulsion;
    std::vector<std::vector<std::vector<std::vector<double>>>> m_repulsionIntegrals;

public:
    HartreeFockSolver(std::vector<BasisFunction> basis_functions,
        std::vector<BasisFunction> basis_gradients, std::vector<std::pair<vec3, double>> nuclei,
        const unsigned int number_of_electrons);

    void guessInitialDensity();
    void calcKineticEnergy();
    void calcPotential();

    void calcElectronRepulsion();
    void calcRepulsionIntegrals();

    void calcOverlap();

    double solveSCF();
};

};
