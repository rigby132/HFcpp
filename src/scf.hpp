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

#include "basis.hpp"

#include <Eigen/Dense>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace hf {

using Matrix = Eigen::MatrixXd;
using Repulsions = std::vector<std::vector<std::vector<std::vector<double>>>>;

class HFSolver {
private:
    Matrix calcDensity(const Matrix& coeff);

    Matrix calcOverlap();
    Matrix calcKineticEnergy();
    Matrix calcPotential(size_t nucleus);

    Repulsions calcRepulsionIntegrals();
    Matrix calcElectronRepulsion(const Repulsions& integrals, const Matrix& density);

public:
    /**
     * @brief Construct a new HFSolver object
     *
     * @param basis The basis functions to use.
     * @param nuclei The nuclei positions and charges.
     * @param occupied_orbitals Number of fully occupied orbitals, starting from lowest one.
     */
    HFSolver(const std::vector<CGTO<double>>& basis, const std::vector<Nucleus<>>& nuclei,
        const unsigned int occupied_orbitals);

    /**
     * @brief Does the complete Hartree-Fock calculation on the provided structure.
     *
     * @param tolerance The maximum absolute difference between iterations before the value is
     * considered converged.
     *
     * @return double The ground-state Energy of the provided atomic structure.
     */
    double solve(double tolerance);

    /**
     * @brief Evaluates the a single orbital wavefunction.
     *
     * @param x x coordinate.
     * @param y y coordinate.
     * @param z z coordinate.
     * @param level The index of the orbital in order of the calculated energy levels.
     * @return The value at the specified point of the orbial.
     */
    double orbital(double x, double y, double z, unsigned int level) const;

public:
    const std::vector<CGTO<double>> m_basis;

    /**
     * @brief Number of basis functions used.
     */
    const unsigned int m_basisSize;

    const std::vector<Nucleus<>> m_nuclei;
    /**
     * @brief Number of occupied orbitals starting from the lowest energy one.
     *
     */
    const unsigned int m_occupied;

private:
    Matrix m_coeff;

};
}