/**
 * @file scf.cpp
 * @author Deniz Güven (s0394473@uni-frankfrut.de)
 * @brief Defines various functions for the scf method.
 * @version 0.1
 * @date 2021-07-11
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "scf.hpp"
#include "basis.hpp"
#include "boys.hpp"

#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

// TODO REMOVE!
#include <cstdio>
#include <iomanip>
#include <iostream>

constexpr double PI = 3.141592653589793;

hf::HFSolver::HFSolver(const std::vector<CGTO<double>>& basis, const std::vector<Nucleus<>>& nuclei,
    const int occupied_orbitals)
    : m_basis(basis)
    , m_basisSize(static_cast<int>(m_basis.size()))
    , m_nuclei(nuclei)
    , m_occupied(occupied_orbitals)
    , m_coeff(m_basisSize, m_basisSize)
{
    m_coeff.setZero();

    if (m_occupied > m_basisSize)
        throw std::invalid_argument(
            "Not enough basis functions for specified amount of occupied orbitals");

    // TODO other checks.
}

hf::Matrix hf::HFSolver::calcDensity(const Matrix& coeff)
{
    Matrix density(m_basisSize, m_basisSize);
    density.setZero();
    for (int t = 0; t < m_basisSize; t++)
        for (int u = 0; u < m_basisSize; u++)
            for (int i = 0; i < m_occupied; i++)
                density(t, u) += 2.0 * coeff(t, i) * coeff(u, i);

    return density;
}

template <typename FLOAT = double>
FLOAT calcSingleOverlap(
    const FLOAT a, const FLOAT b, const FLOAT a_x, FLOAT b_x, const size_t qa, const size_t qb)
{
    // Some useful constants
    const FLOAT p = a + b;
    const FLOAT mu = a * b / p;
    // const FLOAT p_x = (a * a_x + b * b_x) / p;
    const FLOAT x_ab = a_x - b_x;
    const FLOAT k_ab = std::exp(-mu * x_ab * x_ab);
    const FLOAT x_pa = -b * x_ab / p;
    const FLOAT x_pb = a * x_ab / p;

    // Overlap integrals (over product of 2 GTOs) for all quantum numbers up to S_ij.
    constexpr auto nan = std::numeric_limits<FLOAT>::quiet_NaN();

    std::vector<std::vector<FLOAT>> S(
        qa >= 2 ? qa + 1 : 2, std::vector<FLOAT>(qb >= 2 ? qb + 1 : 2, nan));
    S[0][0] = std::sqrt(PI / p) * k_ab;
    S[1][0] = x_pa * S[0][0];
    S[0][1] = x_pb * S[0][0];
    S[1][1] = x_pa * S[0][1] + 0.5 * S[0][0] / p;

    if (qa >= qb) {
        // Fill S with values for all i  and j = 0 and 1.
        for (size_t i = 2; i <= qa; i++) {
            S[i][0] = x_pa * S[i - 1][0] + (0.5 / p) * (i - 1.0) * S[i - 2][0];
            S[i][1] = x_pa * S[i - 1][1] + (0.5 / p) * ((i - 1.0) * S[i - 2][1] + S[i - 1][0]);
        }

        // Move up to reach j = qb by recursion.
        for (size_t j = 2; j <= qb; j++)
            for (size_t i = qa - qb + j; i <= qa; i++)
                S[i][j] = x_pb * S[i][j - 1]
                    + (i * S[i - 1][j - 1] + (j - 1.0) * S[i][j - 2]) / (2 * p);
    } else {
        // Fill S with values for all j  and i = 0 and 1.
        for (size_t j = 2; j <= qb; j++) {
            S[0][j] = x_pb * S[0][j - 1] + (0.5 / p) * (j - 1.0) * S[0][j - 2];
            S[1][j] = x_pb * S[1][j - 1] + (0.5 / p) * ((j - 1.0) * S[1][j - 2] + S[0][j - 1]);
        }

        // Move up to reach i = qa by recursion.
        for (size_t i = 2; i <= qa; i++)
            for (size_t j = qb - qa + i; j <= qb; j++)
                S[i][j] = x_pa * S[i - 1][j]
                    + ((i - 1.0) * S[i - 2][j] + j * S[i - 1][j - 1]) / (2 * p);
    }

    return S[qa][qb];
}

hf::Matrix hf::HFSolver::calcOverlap()
{
    Matrix overlap(m_basisSize, m_basisSize);

    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++) {
            const auto& cg0 = m_basis[r];
            const auto& cg1 = m_basis[s];

            double sum = 0.0;
            // Calc overlap between cgto0 and cgto1
            for (const auto& g0 : cg0.gtos_)
                for (const auto& g1 : cg1.gtos_)
                    sum += g0.c_ * g1.c_
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cx_, cg1.cx_, g0.i_, g1.i_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cy_, cg1.cy_, g0.j_, g1.j_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cz_, cg1.cz_, g0.k_, g1.k_);

            assert(!std::isnan(sum));
            overlap(r, s) = sum;
        }

    return overlap;
}

/**
 * @brief Calculates a single kin-energy integral for the specified primitive GTOs.
 *
 * @tparam FLOAT The floating point type to use.
 * @param a The exponent of GTO a.
 * @param b The exponent of GTO b.
 * @param a_x The center coordinate of GTO a.
 * @param b_x The center coordinate of GTO b.
 * @param qa The orbital anulgar momentum quantum number of GTO a.
 * @param qb The orbital anulgar momentum quantum number of GTO b.
 *
 * @return The Integral of GTOa * GTOb over -inf & +inf.
 */
template <typename FLOAT = double>
FLOAT calcSingleKinEnergy(const FLOAT a, const FLOAT b, const FLOAT a_x, const FLOAT b_x,
    const size_t qa, const size_t qb)
{
    FLOAT d = 4 * a * a * calcSingleOverlap(a, b, a_x, b_x, qa + 2, qb)
        - 2 * a * (2.0 * qa + 1.0) * calcSingleOverlap(a, b, a_x, b_x, qa, qb);
    if (qa > 1)
        d += qa * (qa - 1.0) * calcSingleOverlap(a, b, a_x, b_x, qa - 2, qb);

    return d;
}

hf::Matrix hf::HFSolver::calcKineticEnergy()
{
    Matrix kinEnergy(m_basisSize, m_basisSize);

    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++) {
            const auto& cg0 = m_basis[r];
            const auto& cg1 = m_basis[s];

            double sum = 0;
            // Calc overlap between cgto0 and cgto1
            for (const auto& g0 : cg0.gtos_)
                for (const auto& g1 : cg1.gtos_) {
                    sum += -0.5 * g0.c_ * g1.c_
                        * calcSingleKinEnergy(g0.a_, g1.a_, cg0.cx_, cg1.cx_, g0.i_, g1.i_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cy_, cg1.cy_, g0.j_, g1.j_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cz_, cg1.cz_, g0.k_, g1.k_);

                    sum += -0.5 * g0.c_ * g1.c_
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cx_, cg1.cx_, g0.i_, g1.i_)
                        * calcSingleKinEnergy(g0.a_, g1.a_, cg0.cy_, cg1.cy_, g0.j_, g1.j_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cz_, cg1.cz_, g0.k_, g1.k_);

                    sum += -0.5 * g0.c_ * g1.c_
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cx_, cg1.cx_, g0.i_, g1.i_)
                        * calcSingleOverlap(g0.a_, g1.a_, cg0.cy_, cg1.cy_, g0.j_, g1.j_)
                        * calcSingleKinEnergy(g0.a_, g1.a_, cg0.cz_, cg1.cz_, g0.k_, g1.k_);
                }

            assert(!std::isnan(sum));
            kinEnergy(r, s) = sum;
        }

    return kinEnergy;
}

template <typename FLOAT = double>
std::vector<FLOAT> calcPotentialIntegral(const std::vector<FLOAT>& start, const int qa,
    const int qb, const FLOAT x_pa, const FLOAT x_pz, const FLOAT x_ab, const FLOAT p)
{
    assert(start.size() > 0);

    const int N = static_cast<int>(start.size()) - 1;
    const int qn = qa + qb; // Calculate up to i+1

    assert(N >= qn);

    constexpr auto nan = std::numeric_limits<FLOAT>::quiet_NaN();

    // Create single 3d array for all recursion elements
    std::vector<std::vector<std::vector<FLOAT>>> theta(
        qn + 1, std::vector<std::vector<FLOAT>>(qb + 1, std::vector<FLOAT>(N + 1, nan)));

    // Set initial values.
    theta[0][0] = start;

    // Calculate all thetas[i][0][n] up to i=qn and n = N
    for (int i = 1; i <= qn; i++) {
        for (int n = N - i; n >= 0; n--) {
            theta[i][0][n] = x_pa * theta[i - 1][0][n] - x_pz * theta[i - 1][0][n + 1];

            if (i > 1)
                theta[i][0][n]
                    += 0.5 * (i - 1.0) * (theta[i - 2][0][n] - theta[i - 2][0][n + 1]) / p;
        }
    }

    // Do horizontal recursion for j.
    for (int j = 1; j <= qb; j++)
        for (int i = qa; i <= qn - j; i++)
            for (int n = 0; n <= N - qn; n++)
                theta[i][j][n] = theta[i + 1][j - 1][n] + x_ab * theta[i][j - 1][n];

    // Return only needed orders of theta(0 to N-qn)
    return std::vector<FLOAT>(theta[qa][qb].begin(), theta[qa][qb].begin() + N - qn + 1);
}

hf::Matrix hf::HFSolver::calcPotential(int i)
{
    hf::Matrix potential(m_basisSize, m_basisSize);
    const auto& nuc = m_nuclei.at(i);

    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++) {

            const CGTO<double>& cg0 = m_basis[r];
            const CGTO<double>& cg1 = m_basis[s];

            const auto x_ab = cg0.cx_ - cg1.cx_;
            const auto y_ab = cg0.cy_ - cg1.cy_;
            const auto z_ab = cg0.cz_ - cg1.cz_;

            double sum = 0;
            for (const GTO<double>& g0 : cg0.gtos_)
                for (const GTO<double>& g1 : cg1.gtos_) {

                    const auto a = g0.a_;
                    const auto b = g1.a_;
                    const auto p = a + b;
                    const auto px = (a * cg0.cx_ + b * cg1.cx_) / p;
                    const auto py = (a * cg0.cy_ + b * cg1.cy_) / p;
                    const auto pz = (a * cg0.cz_ + b * cg1.cz_) / p;
                    const auto mu = a * b / (a + b);
                    const auto k_ab = std::exp(-mu * (x_ab * x_ab + y_ab * y_ab + z_ab * z_ab));

                    const auto sep2 = (px - nuc.x) * (px - nuc.x) + (py - nuc.y) * (py - nuc.y)
                        + (pz - nuc.z) * (pz - nuc.z);

                    const auto N = g0.i_ + g0.j_ + g0.k_ + g1.i_ + g1.j_ + g1.k_;

                    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

                    std::vector<double> potentials(N + 1, nan);
                    for (int n = 0; n <= N; n++)
                        potentials[n] = (2 * PI / p) * k_ab * boys(n, p * sep2);

                    potentials = calcPotentialIntegral(
                        potentials, g0.i_, g1.i_, px - cg0.cx_, px - nuc.x, x_ab, p);
                    potentials = calcPotentialIntegral(
                        potentials, g0.j_, g1.j_, py - cg0.cy_, py - nuc.y, y_ab, p);
                    potentials = calcPotentialIntegral(
                        potentials, g0.k_, g1.k_, pz - cg0.cz_, pz - nuc.z, z_ab, p);

                    sum += g0.c_ * g1.c_ * static_cast<double>(nuc.charge) * potentials[0];
                }
            assert(!std::isnan(sum));

            potential(r, s) = sum;
        }
    return potential;
}

/**
 * @brief Calculates the 2 electron repulsion integral of 4 primitive GTOs.
 *
 * @tparam FLOAT The floating point type to use.
 * @param start The Initial starting values for this dimension(eg. x|y|z).
 * @param quantum An array containing the 4 quantum numbers i,j,k,l (one for each orbital).
 * @param expo An array containing the 4 exponents of each orbital function.
 * @param cx An array containing the center coordinates of each orbital.
 * @return The 2 electron repulsion integral.
 */
template <typename FLOAT = double>
std::vector<FLOAT> calcSingleRepulsionIntegral(const std::vector<FLOAT>& start,
    const int (&quantum)[4], const FLOAT (&expo)[4], const FLOAT (&cx)[4])
{
    // TODO Add FLOAT assert checks everywhere.

    assert(start.size() > 0);

    // The highest order to reach in each recursion.
    const int N = static_cast<int>(start.size()) - 1;

    // The quantum numbers to reach.
    const auto qi = quantum[0];
    const auto qj = quantum[1];
    const auto qk = quantum[2];
    const auto ql = quantum[3];
    const auto qsum = qi + qj + qk + ql;

    // This function uses qi+qj+qk+ql orders up with N provided.
    // => So N must be at least equal to that.
    assert(N >= qsum);

    // Define constants used in the recursions.
    const FLOAT p = expo[0] + expo[1];
    const FLOAT px = (expo[0] * cx[0] + expo[1] * cx[1]) / p;
    const FLOAT q = expo[2] + expo[3];
    const FLOAT qx = (expo[2] * cx[2] + expo[3] * cx[3]) / q;
    const FLOAT x_ab = cx[0] - cx[1];
    const FLOAT x_cd = cx[2] - cx[3];
    const FLOAT x_pa = px - cx[0];
    const FLOAT x_pq = px - qx;
    const FLOAT a = p * q / (p + q);

    const FLOAT nan = std::numeric_limits<FLOAT>::quiet_NaN();

    // First recursion: Get first index up to u+j+k+l(the sum of all quantum numbers) .
    std::vector<std::vector<FLOAT>> theta(qsum + 1);
    theta[0] = start;

    // The first recursion must go up to i = sum(quantum)
    // so enough values are available for the other recursions.
    for (int i = 1; i <= qsum; i++) {
        theta[i] = std::vector<FLOAT>(N - i + 1, nan);
        for (int n = 0; n <= N - i; n++) {
            theta[i][n] = x_pa * theta[i - 1][n] - a * x_pq * theta[i - 1][n + 1] / p;

            if (i >= 2)
                theta[i][n]
                    += (i - 1.0) * (theta[i - 2][n] - a * theta[i - 2][n + 1] / p) / (2 * p);
        }
    }

    // Second recursion:
    std::vector<std::vector<std::vector<FLOAT>>> omega(qk + ql + 1);
    omega[0] = theta;

    for (int k = 1; k <= qk + ql; k++) {
        omega[k] = std::vector<std::vector<FLOAT>>(qsum - k + 1);

        for (int i = 0; i <= qsum - k; i++) {
            omega[k][i] = std::vector<FLOAT>(N - qsum + 1, nan);

            for (int n = 0; n <= N - qsum; n++) {
                const FLOAT factor = -(expo[1] * x_ab + expo[3] * x_cd) / q;
                FLOAT val = factor * omega[k - 1][i][n] - p * omega[k - 1][i + 1][n] / q;

                if (i > 0)
                    val += i * omega[k - 1][i - 1][n] / (2 * q);

                if (k > 1)
                    val += (k - 1.0) * omega[k - 2][i][n] / (2 * q);

                omega[k][i][n] = val;
            }
        }
    }

    // Third recursion:
    std::vector<std::vector<std::vector<std::vector<FLOAT>>>> tau(
        qj + 1, std::vector<std::vector<std::vector<FLOAT>>>(qk + ql + 1));
    tau[0] = omega;

    for (int j = 1; j <= qj; j++)
        for (int k = 0; k <= qk + ql; k++) {
            tau[j][k] = std::vector<std::vector<FLOAT>>(
                qi + qj - j + 1, std::vector<FLOAT>(N - qsum + 1, nan));

            for (int i = 0; i <= qi + qj - j; i++)
                for (int n = 0; n <= N - qsum; n++)
                    tau[j][k][i][n] = tau[j - 1][k][i + 1][n] + x_ab * tau[j - 1][k][i][n];
        }

    // Fourth recursion:
    // Use smaller table because i & j do not need to change anymore.
    std::vector<std::vector<std::vector<FLOAT>>> chi(ql + 1);
    chi[0] = std::vector<std::vector<FLOAT>>(qk + ql + 1);

    // Copy values into smaller vector.
    for (int k = 0; k <= qk + ql; k++) {
        chi[0][k] = std::vector<FLOAT>(N - qsum + 1, nan);
        for (int n = 0; n <= N - qsum; n++) {
            chi[0][k][n] = tau[qj][k][qi][n];
        }
    }

    // Do the transfer recursion calculation.
    for (int l = 1; l <= ql; l++) {
        chi[l] = std::vector<std::vector<FLOAT>>(
            qk + ql - l + 1, std::vector<FLOAT>(N - qsum + 1, nan));

        for (int k = 0; k <= qk + ql - l; k++)
            for (int n = 0; n <= N - qsum; n++)
                chi[l][k][n] = chi[l - 1][k + 1][n] + x_cd * chi[l - 1][k][n];
    }

    return chi[ql][qk];
}

hf::Repulsions hf::HFSolver::calcRepulsionIntegrals()
{
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    Repulsions integrals(static_cast<size_t>(m_basisSize));

    for (auto& lvl1 : integrals) {
        lvl1.resize(static_cast<size_t>(m_basisSize));
        for (auto& lvl2 : lvl1) {
            lvl2.resize(static_cast<size_t>(m_basisSize));
            for (auto& lvl3 : lvl2)
                lvl3.resize(static_cast<size_t>(m_basisSize), nan);
        }
    }

#pragma omp parallel for schedule(guided)
    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++)
            for (int t = 0; t < m_basisSize; t++)
                for (int u = 0; u < m_basisSize; u++) {
                    const auto& cg0 = m_basis[r];
                    const auto& cg1 = m_basis[s];
                    const auto& cg2 = m_basis[t];
                    const auto& cg3 = m_basis[u];

                    double sum = 0;

                    for (const auto& g0 : cg0.gtos_)
                        for (const auto& g1 : cg1.gtos_)
                            for (const auto& g2 : cg2.gtos_)
                                for (const auto& g3 : cg3.gtos_) {
                                    int qsum = g0.i_ + g1.i_ + g2.i_ + g3.i_;
                                    qsum += g0.j_ + g1.j_ + g2.j_ + g3.j_;
                                    qsum += g0.k_ + g1.k_ + g2.k_ + g3.k_;

                                    double p = g0.a_ + g1.a_;
                                    double q = g2.a_ + g3.a_;
                                    double mu_ab = g0.a_ * g1.a_ / p;
                                    double mu_cd = g2.a_ * g3.a_ / q;

                                    double dist2_ab = (cg0.cx_ - cg1.cx_) * (cg0.cx_ - cg1.cx_)
                                        + (cg0.cy_ - cg1.cy_) * (cg0.cy_ - cg1.cy_)
                                        + (cg0.cz_ - cg1.cz_) * (cg0.cz_ - cg1.cz_);

                                    double dist2_cd = (cg2.cx_ - cg3.cx_) * (cg2.cx_ - cg3.cx_)
                                        + (cg2.cy_ - cg3.cy_) * (cg2.cy_ - cg3.cy_)
                                        + (cg2.cz_ - cg3.cz_) * (cg2.cz_ - cg3.cz_);

                                    double k_ab = std::exp(-mu_ab * dist2_ab);
                                    double k_cd = std::exp(-mu_cd * dist2_cd);

                                    double f
                                        = 2 * std::pow(PI, 5.0 / 2.0) / (p * q * std::sqrt(p + q));

                                    const double px = (g0.a_ * cg0.cx_ + g1.a_ * cg1.cx_) / p;
                                    const double py = (g0.a_ * cg0.cy_ + g1.a_ * cg1.cy_) / p;
                                    const double pz = (g0.a_ * cg0.cz_ + g1.a_ * cg1.cz_) / p;
                                    const double qx = (g2.a_ * cg2.cx_ + g3.a_ * cg3.cx_) / q;
                                    const double qy = (g2.a_ * cg2.cy_ + g3.a_ * cg3.cy_) / q;
                                    const double qz = (g2.a_ * cg2.cz_ + g3.a_ * cg3.cz_) / q;

                                    const double dist2_pq = (px - qx) * (px - qx)
                                        + (py - qy) * (py - qy) + (pz - qz) * (pz - qz);

                                    std::vector<double> integral(qsum + 1, nan);

                                    for (int n = 0; n <= qsum; n++)
                                        integral[n]
                                            = f * k_ab * k_cd * boys(n, p * q * dist2_pq / (p + q));

                                    integral = calcSingleRepulsionIntegral(integral,
                                        { g0.i_, g1.i_, g2.i_, g3.i_ },
                                        { g0.a_, g1.a_, g2.a_, g3.a_ },
                                        { cg0.cx_, cg1.cx_, cg2.cx_, cg3.cx_ });

                                    integral = calcSingleRepulsionIntegral(integral,
                                        { g0.j_, g1.j_, g2.j_, g3.j_ },
                                        { g0.a_, g1.a_, g2.a_, g3.a_ },
                                        { cg0.cy_, cg1.cy_, cg2.cy_, cg3.cy_ });

                                    integral = calcSingleRepulsionIntegral(integral,
                                        { g0.k_, g1.k_, g2.k_, g3.k_ },
                                        { g0.a_, g1.a_, g2.a_, g3.a_ },
                                        { cg0.cz_, cg1.cz_, cg2.cz_, cg3.cz_ });

                                    assert(integral.size() == 1);

                                    sum += g0.c_ * g1.c_ * g2.c_ * g3.c_ * integral[0];
                                }

                    assert(!std::isnan(sum));

#pragma omp critical
                    integrals[r][s][t][u] = sum;
                }

    return integrals;
}

hf::Matrix hf::HFSolver::calcElectronRepulsion(
    const hf::Repulsions& integrals, const Matrix& density)
{
    hf::Matrix repulsion(m_basisSize, m_basisSize);
    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++) {

            double sum = 0.0;

            for (int t = 0; t < m_basisSize; t++)
                for (int u = 0; u < m_basisSize; u++)
                    sum += density(t, u) * (integrals[r][s][t][u] - 0.5 * integrals[r][u][t][s]);

            repulsion(r, s) = sum;
            assert(!std::isnan(sum));
        }

    return repulsion;
}

double hf::HFSolver::solve(double tolerance)
{
    auto overlap = calcOverlap();
    // std::cout << "OVERLAP:\n" << overlap << '\n';
    auto kinEnergy = calcKineticEnergy();

    std::vector<Matrix> potentials(m_nuclei.size());
    for (int i = 0; i < static_cast<int>(potentials.size()); i++)
        potentials[i] = calcPotential(i);

    Matrix hcore = kinEnergy;
    for (const auto& potential : potentials)
        hcore -= potential;

    // Matrix density = guessInitialDensity(hcore);
    Matrix density(m_basisSize, m_basisSize);
    density.setZero(); // Hcore initial guess.

    // Diagonalize overlap
    Eigen::SelfAdjointEigenSolver<Matrix> solver(overlap, Eigen::ComputeEigenvectors);
    Matrix D = solver.eigenvalues().asDiagonal();
    Matrix P = solver.eigenvectors();

    // Remove eigenvalues smaller than 10e-5 (they cause numerical problems).
    // Find first problematic index starting from greatest eigenvalue.
    int valuesToRemove = 0;
    for (int i = 0; i < m_basisSize; i++)
        if (D(i, i) < 10e-5)
            valuesToRemove++;
    std::cout << valuesToRemove << " basis functions were removed for numerical stability.\n";

    // Abort if true;
    if (valuesToRemove >= m_basisSize or m_basisSize - valuesToRemove < m_occupied) {
        std::cout << "Too basis functions were removed due to overlap. Aborting...\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Remove all eigenvalues/eigenvectors starting at that index.
    // Reshape matrices to new dimensions and fill values:
    Matrix Dtrunc(m_basisSize - valuesToRemove, m_basisSize);
    for (int i = valuesToRemove; i < m_basisSize; i++)
        for (int j = 0; j < m_basisSize; j++)
            Dtrunc(i - valuesToRemove, j) = D(i, j);

    Matrix Ptrunc(m_basisSize, m_basisSize - valuesToRemove);
    for (int i = 0; i < m_basisSize; i++)
        for (int j = valuesToRemove; j < m_basisSize; j++)
            Ptrunc(i, j - valuesToRemove) = P(i, j);

    std::cout << "Overlap eigenvalues: \n";
    for (int i = 0; i < m_basisSize; i++)
        std::cout << D(i, i) << '\n';
    std::cout << '\n';

    //std::cout << "Truncated overlap matrices:\n" << Dtrunc << "\n\n" << Ptrunc;
    // Use truncated matrices from now on. 
    D = Dtrunc;
    P = Ptrunc;

    for (int i = 0; i < m_basisSize - valuesToRemove; i++)
        D(i, i + valuesToRemove) = 1.0 / std::sqrt(std::abs(D(i, i + valuesToRemove)));

    // Matrix overlap_inv = P * D * P.adjoint();
    Matrix transform = P * D;
    // std::cout << "OVERLAP_INV:\n" << overlap_inv << '\n';
    //std::cout << "TRANSFORM:\n" << transform << '\n';

    auto repulsionIntegrals = calcRepulsionIntegrals();

    double maxDelta;
    int iterations = 0;

    Matrix prevFock = calcElectronRepulsion(repulsionIntegrals, density) + hcore;

    do {
        // std::cout << "ITERATION " << iterations << ":\n";

        auto repulsions = calcElectronRepulsion(repulsionIntegrals, density);

        // std::cout << "REPULSIONS:\n" << repulsions << '\n';

        Matrix fockOperator = repulsions + hcore;

        prevFock = fockOperator;

        // fockOperator = overlap_inv * fockOperator * overlap_inv;
        fockOperator = transform.adjoint() * fockOperator * transform;

        solver.compute(fockOperator);

        // m_coeff = overlap_inv * solver.eigenvectors();
        m_coeff = transform * solver.eigenvectors();

        /*std::cout << "ENERGIES:\n";
        for (int i = 0; i < m_basisSize; i++)
            std::cout << solver.eigenvalues()(i) << '\n';
        std::cout << '\n';*/

        // Update density matrix.
        auto prevDensity = density;
        density = 0.8 * calcDensity(m_coeff) + 0.2 * prevDensity;
        maxDelta = (prevDensity - density).cwiseAbs().maxCoeff();

        iterations++;
    } while (maxDelta > tolerance && iterations < 1000);

    if (iterations >= 1000)
        std::cout << "CALCULATION DID NOT CONVERGE!\n";
    else
        std::cout << "CALCULATION DID CONVERGE!\n";

    double HFEnergy = 0;
    for (int i = 0; i < m_occupied; i++)
        HFEnergy += solver.eigenvalues()(i);

    std::cout << "E-Levels:\n" << solver.eigenvalues() << '\n';

    for (int r = 0; r < m_basisSize; r++)
        for (int s = 0; s < m_basisSize; s++)
            HFEnergy += 0.5 * density(r, s) * hcore(r, s);

    std::cout << "HF-Energy = " << HFEnergy << '\n';

    // Inter-nuclear repulsion energy
    double INREnergy = 0;
    for (unsigned int i = 0; i < m_nuclei.size(); i++)
        for (unsigned int j = i + 1; j < m_nuclei.size(); j++) {
            const auto& nuc0 = m_nuclei[i];
            const auto& nuc1 = m_nuclei[j];

            INREnergy += static_cast<double>(nuc0.charge) * static_cast<double>(nuc1.charge)
                / std::sqrt((nuc0.x - nuc1.x) * (nuc0.x - nuc1.x)
                    + (nuc0.y - nuc1.y) * (nuc0.y - nuc1.y)
                    + (nuc0.z - nuc1.z) * (nuc0.z - nuc1.z));
        }
    std::cout << "INR-Energy = " << INREnergy << '\n';
    std::cout << "Total energy = " << HFEnergy + INREnergy << '\n';

    return HFEnergy + INREnergy;
}

double hf::HFSolver::orbital(double x, double y, double z, size_t n) const
{
    assert(n <= static_cast<size_t>(m_basisSize));

    double sum = 0;
    for (int i = 0; i < m_basisSize; i++)
        sum += m_coeff(i, n) * m_basis[i](x, y, z);

    return sum;
}