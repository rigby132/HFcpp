/**
 * @file scf.hpp
 * @author Deniz Güven (s0394473@uni-frankfrut.de)
 * @brief Implements a hartree-fock solver.
 * @version 0.1
 * @date 2021-03-15
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <Eigen/Dense>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace hf {

template <size_t N, typename T = double> using BasisFunction = std::function<T(std::array<T, N>)>;
/**
 * @brief Information of the Nucleus: Position[0-2] and charge[3]
 *
 * @tparam T Type of number.
 */
template <typename T = double> struct Nucleus {
    T x, y, z;
    T charge;
};
using Matrix = Eigen::MatrixXd;
using Repulsions = std::vector<std::vector<std::vector<std::vector<double>>>>;

class HFSolver {
private:
    const std::vector<BasisFunction<3>> m_basisFunctions;
    const std::vector<BasisFunction<3>> m_basisGradients;
    /**
     * @brief Number of basis functions used.
     */
    const unsigned int m_basisSize;

    const std::vector<Nucleus<>> m_nuclei;
    const unsigned int m_numElectrons;
    /**
     * @brief Number of occupied orbitals starting from the lowest energy one.
     * 
     */
    const unsigned int m_occupied;

    const unsigned long m_sampleSize = 1000000000;

    /**
     * @brief Contains condensed coefficients of each basisfunction, which make up the whole
     * wavefunction.
     */
    Matrix m_density;

    Matrix guessInitialDensity();

    Matrix calcOverlap();
    Matrix calcKineticEnergy();
    Matrix calcPotential(size_t nucleus);

    Repulsions calcRepulsionIntegrals();
    Matrix calcElectronRepulsion(const Repulsions& integrals, const Matrix& density);

public:
    /**
     * @brief Construct a new HFSolver object
     *
     * @param basisFunctions The basis functions to use.
     * @param basisGradients The corresponding gradients for each basis function.
     * @param nuclei The nuclei positions and charges.
     * @param numElectrons Number of electrons contained in the structure.
     */
    HFSolver(const std::vector<BasisFunction<3>>& basisFunctions,
        const std::vector<BasisFunction<3>>& basisGradients, const std::vector<Nucleus<>>& nuclei,
        const unsigned int numElectrons);

    /**
     * @brief Does the complete Hartree-Fock calculation on the provided structure.
     *
     * @param tolerance The maximum numeric difference between iterations before the value is
     * considered converged.
     *
     * @return double The ground-state Energy of the provided atomic structure.
     */
    double solve(double tolerance);
};
}