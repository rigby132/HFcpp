#include "scf.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <omp.h>

#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>

scf::HartreeFockSolver::HartreeFockSolver(std::vector<BasisFunction> basis_functions,
    std::vector<BasisFunction> basis_gradients, std::vector<std::pair<vec3, double>> nuclei,
    const unsigned int number_of_electrons)
    : m_basisFunctions(basis_functions)
    , m_basisGradients(basis_gradients)
    , m_nuclei(nuclei)
    , m_size(basis_functions.size())
    , m_electrons(number_of_electrons)
    , m_density(m_electrons, m_electrons)
    , m_overlap(m_size, m_size)
    , m_kinEnergy(m_size, m_size)
    , m_potEnergies(m_nuclei.size(), Eigen::MatrixXd(m_size, m_size))
    , m_electronRepulsion(m_size, m_size)
    , m_repulsionIntegrals(m_size)
{
    assert(m_basisFunctions.size() == m_basisGradients.size());

    auto size = m_basisFunctions.size();

    for (auto& lvl1 : m_repulsionIntegrals) {
        lvl1.resize(size);
        for (auto& lvl2 : lvl1) {
            lvl2.resize(size);
            for (auto& lvl3 : lvl2)
                lvl3.resize(size);
        }
    }
}

void scf::HartreeFockSolver::calcKineticEnergy()
{
#pragma omp parallel for collapse(2)
    for (unsigned int r = 0; r < m_basisFunctions.size(); r++)
        for (unsigned int s = 0; s < m_basisFunctions.size(); s++) {

            auto fn = [=](const std::vector<double>& values) -> double {
                return m_basisFunctions[r]({ values[0], values[1], values[2] })
                    * m_basisGradients[s]({ values[0], values[1], values[2] });
            };

            /*           m_kinEnergy(r, s) = -0.5
                           * mc_integrator(fn, m_sampleSize, 3,
                               { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound,
               m_lowerBound, m_upperBound });
           */
            Eigen::VectorXd mid(3);
            mid(0) = 0.5 * (m_nuclei[0].first.x + m_nuclei[1].first.x);
            mid(1) = 0.5 * (m_nuclei[0].first.y + m_nuclei[1].first.y);
            mid(2) = 0.5 * (m_nuclei[0].first.z + m_nuclei[1].first.z);

            Eigen::VectorXd var(3);
            var(0) = 2;
            var(1) = 1;
            var(2) = 1;

            m_kinEnergy(r, s) = -0.5 * mc_normal_int(fn, m_sampleSize, 3, var, mid);
        }
    std::cout << "E_kin =\n" << m_kinEnergy << std::endl << std::endl;
}

void scf::HartreeFockSolver::calcPotential()
{
#pragma omp parallel for collapse(3)
    for (unsigned int i = 0; i < m_nuclei.size(); i++)
        for (unsigned int r = 0; r < m_basisFunctions.size(); r++)
            for (unsigned int s = 0; s < m_basisFunctions.size(); s++) {

                auto fn = [=](const std::vector<double>& values) -> double {
                    double xSquare
                        = (values[0] - m_nuclei[i].first.x) * (values[0] - m_nuclei[i].first.x);
                    double ySquare
                        = (values[1] - m_nuclei[i].first.y) * (values[1] - m_nuclei[i].first.y);
                    double zSquare
                        = (values[2] - m_nuclei[i].first.z) * (values[2] - m_nuclei[i].first.z);
                    double distance = std::sqrt(xSquare + ySquare + zSquare);
                    return m_basisFunctions[r]({ values[0], values[1], values[2] })
                        * m_nuclei[i].second
                        * m_basisFunctions[s]({ values[0], values[1], values[2] }) / distance;
                };

                Eigen::VectorXd mid(3);
                mid(0) = 0.5 * (m_nuclei[0].first.x + m_nuclei[1].first.x);
                mid(1) = 0.5 * (m_nuclei[0].first.y + m_nuclei[1].first.y);
                mid(2) = 0.5 * (m_nuclei[0].first.z + m_nuclei[1].first.z);

                Eigen::VectorXd var(3);
                var(0) = 2;
                var(1) = 1;
                var(2) = 1;

                m_potEnergies[i](r, s) = mc_normal_int(fn, m_sampleSize, 3, var, mid);
                /*m_potEnergies[i](r, s) = mc_integrator(f, m_sampleSize, 3,
                    { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                        m_upperBound });*/
            }
    std::cout << "Potentials:\n";
    for (const auto& potential : m_potEnergies)
        std::cout << potential << std::endl;
    std::cout << std::endl;
}

void scf::HartreeFockSolver::calcRepulsionIntegrals()
{

#pragma omp parallel for collapse(4)
    for (unsigned int r = 0; r < m_size; r++)
        for (unsigned int s = 0; s < m_size; s++)
            for (unsigned int t = 0; t < m_size; t++)
                for (unsigned int u = 0; u < m_size; u++) {
                    auto fn = [=](const std::vector<double>& values) -> double {
                        double xSquare = (values[0] - values[3]) * (values[0] - values[3]);
                        double ySquare = (values[1] - values[4]) * (values[1] - values[4]);
                        double zSquare = (values[2] - values[5]) * (values[2] - values[5]);
                        double distance = std::sqrt(xSquare + ySquare + zSquare);
                        return m_basisFunctions[r]({ values[0], values[1], values[2] })
                            * m_basisFunctions[s]({ values[0], values[1], values[2] })
                            * m_basisFunctions[t]({ values[3], values[4], values[5] })
                            * m_basisFunctions[u]({ values[3], values[4], values[5] }) / distance;
                    };

                    Eigen::VectorXd mid(6);
                    mid(0) = 0.5 * (m_nuclei[0].first.x + m_nuclei[1].first.x);
                    mid(1) = 0.5 * (m_nuclei[0].first.y + m_nuclei[1].first.y);
                    mid(2) = 0.5 * (m_nuclei[0].first.z + m_nuclei[1].first.z);
                    mid(3) = 0.5 * (m_nuclei[0].first.x + m_nuclei[1].first.x);
                    mid(4) = 0.5 * (m_nuclei[0].first.y + m_nuclei[1].first.y);
                    mid(5) = 0.5 * (m_nuclei[0].first.z + m_nuclei[1].first.z);

                    Eigen::VectorXd var(6);
                    var(0) = 1;
                    var(1) = 0.5;
                    var(2) = 0.5;
                    var(3) = 1;
                    var(4) = 0.5;
                    var(5) = 0.5;

                    m_repulsionIntegrals[r][s][t][u] = mc_normal_int(fn, m_sampleSize, 6, var, mid);

                    /*m_repulsionIntegrals[r][s][t][u] = mc_integrator(fn, m_sampleSize, 6,
                        { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                            m_upperBound, m_lowerBound, m_upperBound, m_lowerBound, m_upperBound,
                            m_lowerBound, m_upperBound });*/
                }

    for (unsigned int r = 0; r < m_size; r++)
        for (unsigned int s = 0; s < m_size; s++)
            for (unsigned int t = 0; t < m_size; t++)
                for (unsigned int u = 0; u < m_size; u++)
                    std::cout << "(" << r << ' ' << s << '|' << t << ' ' << u
                              << ") = " << m_repulsionIntegrals[r][s][t][u] << std::endl;

    std::cout << std::endl;
}

void scf::HartreeFockSolver::calcElectronRepulsion()
{
#pragma omp parallel for collapse(2)
    for (unsigned int r = 0; r < m_size; r++)
        for (unsigned int s = 0; s < m_size; s++) {

            double sum = 0.0;

            for (unsigned int t = 0; t < m_electrons; t++)
                for (unsigned int u = 0; u < m_electrons; u++)
                    sum += m_density(t, u)
                        * (m_repulsionIntegrals[r][s][t][u]
                            - 0.5 * m_repulsionIntegrals[r][u][t][s]);

            m_electronRepulsion(r, s) = sum;
        }

    std::cout << "G = \n" << m_electronRepulsion << std::endl << std::endl;
}

void scf::HartreeFockSolver::guessInitialDensity()
{

    // TODO: Do a real approximiation.
    m_density(0, 0) = 0.1240;
    m_density(1, 0) = 0.4318;
    m_density(0, 1) = 0.4318;
    m_density(1, 1) = 1.5034;
}

void scf::HartreeFockSolver::calcOverlap()
{

#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < m_size; i++)
        for (unsigned int j = 0; j < m_size; j++) {
            auto f = [=](const std::vector<double>& values) -> double {
                vec3 pos = { values[0], values[1], values[2] };
                return m_basisFunctions[i](pos) * m_basisFunctions[j](pos);
            };
            m_overlap(i, j) = mc_integrator(f, m_sampleSize, 3,
                { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                    m_upperBound });
        }
}

double scf::HartreeFockSolver::solveSCF()
{
    guessInitialDensity();
    calcOverlap();

    calcKineticEnergy();

    calcPotential();

    calcRepulsionIntegrals();

    Eigen::MatrixXd Hcore = m_kinEnergy;
    for (const auto& potential : m_potEnergies)
        Hcore -= potential; // Potential stabilisiert: Negativer Beitrag!

    // Diagonalize S
    Eigen::MatrixXd S = m_overlap;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(S);
    Eigen::MatrixXd D = solver.eigenvalues().real().asDiagonal();
    Eigen::MatrixXd P = solver.eigenvectors().real();

    for (unsigned int i = 0; i < m_size; i++)
        D(i, i) = 1.0 / std::sqrt(D(i, i));

    Eigen::MatrixXd S_inv = P * D * P.inverse();

    std::cout << "S = \n" << m_overlap << std::endl << std::endl;
    std::cout << "S_inv = \n" << S_inv << std::endl << std::endl;

    for (unsigned int i = 0; i < 10; i++) {

        std::cout << "=============================\n ITERATION : " << i
                  << "\n=============================\n";

        calcElectronRepulsion();

        Eigen::MatrixXd F = m_electronRepulsion + Hcore;

        std::cout << "F = \n" << F << std::endl << std::endl;

        F = S_inv * F * S_inv;

        std::cout << "F' = \n" << F << std::endl << std::endl;

        solver.compute(F);

        std::cout << "Energy levels: \n" << solver.eigenvalues() << std::endl << std::endl;

        Eigen::MatrixXd C_new = S_inv * Eigen::MatrixXd(solver.eigenvectors().real());

        std::cout << "C_new = \n" << C_new << std::endl << std::endl;

        // Update density matrix.
        for (unsigned int i = 0; i < 2; i++)
            for (unsigned int j = 0; j < 2; j++)
                m_density(i, j) = 2.0 * C_new(i, 1) * C_new(j, 1);

        std::cout << "New P=\n" << m_density << std::endl << std::endl;
    }
    return -1;
}
