#include "scf.hpp"

#include <omp.h>

#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
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
    for (unsigned int i = 0; i < dim*2; i += 2)
        volume *= bounds[i + 1] - bounds[i];

    sum *= volume / n;

    return sum;
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

scf::HartreeFockSolver::HartreeFockSolver(std::vector<BasisFunction> basis_functions,
    std::vector<BasisFunction> basis_gradients, std::vector<std::pair<vec3, double>> nuclei)
    : m_basisFunctions(basis_functions)
    , m_basisGradients(basis_gradients)
    , m_nuclei(nuclei)
    , m_size(basis_functions.size())
    , m_kinEnergy(m_basisFunctions.size(), m_basisFunctions.size())
    , m_potEnergies(
          m_nuclei.size(), Eigen::MatrixXd(m_basisFunctions.size(), m_basisFunctions.size()))
    , m_electronRepulsion(m_basisFunctions.size(), m_basisFunctions.size())
    , m_repulsionIntegrals(m_basisFunctions.size())
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

            auto f = [=](const std::vector<double>& values) -> double {
                return m_basisFunctions[r]({ values[0], values[1], values[2] })
                    * m_basisGradients[s]({ values[0], values[1], values[2] });
            };

            m_kinEnergy(r, s) = -0.5
                * mc_integrator(f, m_sampleSize, 3,
                    { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                        m_upperBound });
        }
    std::cout << m_kinEnergy << std::endl << std::endl;
}

void scf::HartreeFockSolver::calcPotential()
{
#pragma omp parallel for collapse(3)
    for (unsigned int i = 0; i < m_nuclei.size(); i++)
        for (unsigned int r = 0; r < m_basisFunctions.size(); r++)
            for (unsigned int s = 0; s < m_basisFunctions.size(); s++) {

                auto f = [=](const std::vector<double>& values) -> double {
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

                m_potEnergies[i](r, s) = mc_integrator(f, m_sampleSize, 3,
                    { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                        m_upperBound });
            }
    for (const auto& potential : m_potEnergies)
        std::cout << potential << std::endl << std::endl;
}

void scf::HartreeFockSolver::calcRepulsionIntegrals()
{

#pragma omp parallel for collapse(4)
    for (unsigned int r = 0; r < m_size; r++)
        for (unsigned int s = 0; s < m_size; s++)
            for (unsigned int t = 0; t < m_size; t++)
                for (unsigned int u = 0; u < m_size; u++) {
                    auto f = [=](const std::vector<double>& values) -> double {
                        double xSquare = (values[0] - values[3]) * (values[0] - values[3]);
                        double ySquare = (values[1] - values[4]) * (values[1] - values[4]);
                        double zSquare = (values[2] - values[5]) * (values[2] - values[5]);
                        double distance = std::sqrt(xSquare + ySquare + zSquare);
                        return m_basisFunctions[r]({ values[0], values[1], values[2] })
                            * m_basisFunctions[s]({ values[0], values[1], values[2] })
                            * m_basisFunctions[t]({ values[3], values[4], values[5] })
                            * m_basisFunctions[u]({ values[3], values[4], values[5] }) / distance;
                    };

                    m_repulsionIntegrals[r][s][t][u] = mc_integrator(f, m_sampleSize, 6,
                        { m_lowerBound, m_upperBound, m_lowerBound, m_upperBound, m_lowerBound,
                            m_upperBound, m_lowerBound, m_upperBound, m_lowerBound, m_upperBound,
                            m_lowerBound, m_upperBound });
                }

    for (unsigned int r = 0; r < m_size; r++)
        for (unsigned int s = 0; s < m_size; s++)
            for (unsigned int t = 0; t < m_size; t++)
                for (unsigned int u = 0; u < m_size; u++)
                    std::cout << "(" << r << ' ' << s << '|' << t << ' ' << u
                              << ") = " << m_repulsionIntegrals[r][s][t][u] << std::endl;

    std::cout << std::endl;
}

double scf::HartreeFockSolver::solveSCF()
{
    calcKineticEnergy();

    calcPotential();

    calcRepulsionIntegrals();

    return -1;
}
