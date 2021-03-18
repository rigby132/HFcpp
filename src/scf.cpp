#include "scf.hpp"
#include "integration.hpp"

#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <cassert>
#include <cstddef>

// TODO REMOVE!
#include <iostream>

hf::HFSolver::HFSolver(const std::vector<BasisFunction<3>>& basisFunctions,
    const std::vector<BasisFunction<3>>& basisGradients, const std::vector<Nucleus<>>& nuclei,
    const unsigned int numElectrons)
    : m_basisFunctions(basisFunctions)
    , m_basisGradients(basisGradients)
    , m_basisSize(m_basisFunctions.size())
    , m_nuclei(nuclei)
    , m_numElectrons(numElectrons)
    , m_occupied(numElectrons / 2)
    , m_density(guessInitialDensity())
{
    // Must obviously match.
    assert(basisFunctions.size() == basisGradients.size());

    // TODO: replace the following 2 asserts with exceptions or reverse definition of electrons with
    // occupied(no exceptions possible)?
    // Only closed shell structures are allowed!
    assert(numElectrons % 2 == 0);

    // If less orbitals than electron-pairs are calculated, then the final energy cannot account for
    // the extra electrons => The final single point energy will be wrong.
    assert(m_basisSize >= numElectrons / 2);
}

hf::Matrix hf::HFSolver::guessInitialDensity()
{
    Matrix density(m_basisSize, m_basisSize);
    // TODO: Do real approximation
    density(0, 0) = 0.1240;
    density(1, 0) = 0.4318;
    density(0, 1) = 0.4318;
    density(1, 1) = 1.5034;

    return density;
}

hf::Matrix hf::HFSolver::calcOverlap()
{
    Matrix overlap(m_basisSize, m_basisSize);

#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < m_basisSize; i++)
        for (unsigned int j = 0; j < m_basisSize; j++) {
            auto fn = [=](const std::array<double, 3>& pos) -> double {
                return m_basisFunctions[i](pos) * m_basisFunctions[j](pos);
            };

            // TODO: Basisfunctions should encode their own position -> maybe basisfunctions as
            // generic functors?
            NormalDistribution<3> dist({ 0, 0, 0 }, { 2, 2, 2 });

            overlap(i, j) = mc_integrate(fn, dist, m_sampleSize);
        }

    return overlap;
}

hf::Matrix hf::HFSolver::calcKineticEnergy()
{
    hf::Matrix kinEnergy(m_basisSize, m_basisSize);
#pragma omp parallel for collapse(2)
    for (unsigned int r = 0; r < m_basisFunctions.size(); r++)
        for (unsigned int s = 0; s < m_basisFunctions.size(); s++) {

            auto fn = [=](const std::array<double, 3>& pos) -> double {
                return m_basisFunctions[r](pos) * m_basisGradients[s](pos);
            };

            std::array<double, 3> mean { 0.5 * (m_nuclei[0].x + m_nuclei[1].x),
                0.5 * (m_nuclei[0].y + m_nuclei[1].y), 0.5 * (m_nuclei[0].z + m_nuclei[1].z) };

            // TODO: Orient deviation on nucleus-nucleus axis.
            NormalDistribution<3> dist(mean, { 2, 1, 1 });

            kinEnergy(r, s) = -0.5 * mc_integrate(fn, dist, m_sampleSize);
        }
    return kinEnergy;
}

hf::Matrix hf::HFSolver::calcPotential(size_t i)
{
    hf::Matrix potential(m_basisSize, m_basisSize);
#pragma omp parallel for collapse(2)
    for (unsigned int r = 0; r < m_basisFunctions.size(); r++)
        for (unsigned int s = 0; s < m_basisFunctions.size(); s++) {

            auto fn = [=](const std::array<double, 3>& values) -> double {
                double xSquare = (values[0] - m_nuclei[i].x) * (values[0] - m_nuclei[i].x);
                double ySquare = (values[1] - m_nuclei[i].y) * (values[1] - m_nuclei[i].y);
                double zSquare = (values[2] - m_nuclei[i].z) * (values[2] - m_nuclei[i].z);
                double distance = std::sqrt(xSquare + ySquare + zSquare);
                return m_basisFunctions[r](values) * m_nuclei[i].charge
                    * m_basisFunctions[s](values) / distance;
            };

            std::array<double, 3> mean { 0.5 * (m_nuclei[0].x + m_nuclei[1].x),
                0.5 * (m_nuclei[0].y + m_nuclei[1].y), 0.5 * (m_nuclei[0].z + m_nuclei[1].z) };

            // TODO: Orient deviation on nucleus-nucleus axis.
            NormalDistribution<3> dist(mean, { 2, 1, 1 });

            potential(r, s) = mc_integrate(fn, dist, m_sampleSize);
        }
    return potential;
}

hf::Repulsions hf::HFSolver::calcRepulsionIntegrals()
{
    Repulsions integrals(m_basisSize);

    for (auto& lvl1 : integrals) {
        lvl1.resize(m_basisSize);
        for (auto& lvl2 : lvl1) {
            lvl2.resize(m_basisSize);
            for (auto& lvl3 : lvl2)
                lvl3.resize(m_basisSize);
        }
    }

#pragma omp parallel for collapse(4)
    for (unsigned int r = 0; r < m_basisSize; r++)
        for (unsigned int s = 0; s < m_basisSize; s++)
            for (unsigned int t = 0; t < m_basisSize; t++)
                for (unsigned int u = 0; u < m_basisSize; u++) {
                    auto fn = [=](const std::array<double, 6>& values) -> double {
                        double xSquare = (values[0] - values[3]) * (values[0] - values[3]);
                        double ySquare = (values[1] - values[4]) * (values[1] - values[4]);
                        double zSquare = (values[2] - values[5]) * (values[2] - values[5]);
                        double distance = std::sqrt(xSquare + ySquare + zSquare);
                        return m_basisFunctions[r]({ values[0], values[1], values[2] })
                            * m_basisFunctions[s]({ values[0], values[1], values[2] })
                            * m_basisFunctions[t]({ values[3], values[4], values[5] })
                            * m_basisFunctions[u]({ values[3], values[4], values[5] }) / distance;
                    };

                    std::array<double, 6> mean { 0.5 * (m_nuclei[0].x + m_nuclei[1].x),
                        0.5 * (m_nuclei[0].y + m_nuclei[1].y),
                        0.5 * (m_nuclei[0].z + m_nuclei[1].z),
                        0.5 * (m_nuclei[0].x + m_nuclei[1].x),
                        0.5 * (m_nuclei[0].y + m_nuclei[1].y),
                        0.5 * (m_nuclei[0].z + m_nuclei[1].z) };

                    NormalDistribution<6> dist(mean, { 2, 1, 1, 2, 1, 1 });

                    integrals[r][s][t][u] = mc_integrate<6>(fn, dist, m_sampleSize);
#pragma omp critical
                    std::cout << '(' << r << ' ' << s << '|' << t << ' ' << u
                              << ") = " << integrals[r][s][t][u] << '\n';
                }
    return integrals;
}

hf::Matrix hf::HFSolver::calcElectronRepulsion(
    const hf::Repulsions& integrals, const Matrix& density)
{
    hf::Matrix repulsion(m_basisSize, m_basisSize);
#pragma omp parallel for collapse(2)
    for (unsigned int r = 0; r < m_basisSize; r++)
        for (unsigned int s = 0; s < m_basisSize; s++) {

            double sum = 0.0;

            for (unsigned int t = 0; t < m_basisSize; t++)
                for (unsigned int u = 0; u < m_basisSize; u++)
                    sum += density(t, u) * (integrals[r][s][t][u] - 0.5 * integrals[r][u][t][s]);

            repulsion(r, s) = sum;
        }

    return repulsion;
}

double hf::HFSolver::solve(double tolerance)
{
    auto overlap = calcOverlap();

    std::cout << "Overlap = \n" << overlap << '\n';

    auto kinEnergy = calcKineticEnergy();

    std::cout << "kinEnergy = \n" << kinEnergy << '\n';

    std::vector<Matrix> potentials(m_basisSize);
    for (unsigned int i = 0; i < potentials.size(); i++)
        potentials[i] = calcPotential(i);

    for (const auto& p : potentials)
        std::cout << "p = \n" << p << '\n';

    auto repulsionIntegrals = calcRepulsionIntegrals();

    Matrix hcore = kinEnergy;
    for (const auto& potential : potentials)
        hcore -= potential;

    // Diagonalize overlap
    Eigen::SelfAdjointEigenSolver<Matrix> solver(overlap, Eigen::ComputeEigenvectors);
    Matrix D = solver.eigenvalues().asDiagonal();
    Matrix P = solver.eigenvectors();

    for (unsigned int i = 0; i < m_basisSize; i++)
        D(i, i) = 1.0 / std::sqrt(D(i, i));

    Matrix overlap_inv = P * D * P.inverse();

    std::cout << "STARTING SCF\n\n";

    double maxDelta;
    unsigned int i = 0;
    do {
        std::cout << "ITERATION: " << i << '\n';
        i++;

        auto repulsions = calcElectronRepulsion(repulsionIntegrals, m_density);

        Matrix fockOperator = repulsions + hcore;

        std::cout << "F = \n" << fockOperator << '\n';

        fockOperator = overlap_inv * fockOperator * overlap_inv;

        solver.compute(fockOperator);

        Matrix coeff = overlap_inv * solver.eigenvectors();

        std::cout << "Coeff =\n" << coeff(0, 1) << '\n';

        // Update density matrix.
        auto prevDensity = m_density;
        m_density.setZero();
        for (unsigned int t = 0; t < m_basisSize; t++)
            for (unsigned int u = 0; u < m_basisSize; u++)
                for (unsigned int i = 0; i < m_occupied; i++)
                    m_density(t, u) += 2.0 * coeff(t, i) * coeff(u, i);

        maxDelta = (prevDensity - m_density).cwiseAbs().maxCoeff();

        std::cout << "density = \n" << m_density << '\n';
        std::cout << "Max delta density = " << maxDelta << '\n' << "\n\n";
    } while (maxDelta > tolerance);

    double HFEnergy = 0;
    for (unsigned int i = 0; i < m_occupied; i++)
        HFEnergy += solver.eigenvalues()(i);

    for (unsigned int r = 0; r < m_basisSize; r++)
        for (unsigned int s = 0; s < m_basisSize; s++)
            HFEnergy += 0.5 * m_density(r, s) * hcore(r, s);

    std::cout << "HF-Energy = " << HFEnergy << '\n';

    // Internuclear repulsion energy
    double IREnergy = 0;
    for (unsigned int i = 0; i < m_nuclei.size(); i++)
        for (unsigned int j = i+1; j < m_nuclei.size(); j++){
            const auto& nuc0 = m_nuclei[i];
            const auto& nuc1 = m_nuclei[j];

            IREnergy += nuc0.charge * nuc1.charge
                / std::sqrt((nuc0.x - nuc1.x) * (nuc0.x - nuc1.x)
                    + (nuc0.y - nuc1.y) * (nuc0.y - nuc1.y)
                    + (nuc0.z - nuc1.z) * (nuc0.z - nuc1.z));
        }
    std::cout << "IR-Energy = " << IREnergy << '\n';
    std::cout << "Total energy = " << HFEnergy + IREnergy << '\n';

    return HFEnergy + IREnergy;
}