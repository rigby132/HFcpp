#include "scf.hpp"

#include <Eigen/Dense>

#include <functional>
#include <iostream>

int main()
{
    std::cout << "scf\n";

    std::vector<scf::BasisFunction> basis
        = { std::bind(scf::gto, std::placeholders::_1, scf::vec3{ 0, 0, 0 }, 0.3696, 0.41660),
              std::bind(scf::gto, std::placeholders::_1, scf::vec3{ 1.5117, 0, 0 }, 0.5881, 0.77390) };

    std::vector<scf::BasisFunction> gradients = {
        std::bind(&scf::gto_gradient, std::placeholders::_1, scf::vec3{ 0, 0, 0 }, 0.3696, 0.41660),
        std::bind(&scf::gto_gradient, std::placeholders::_1, scf::vec3{ 1.5117, 0, 0 }, 0.5881, 0.77390)
    };

    scf::HartreeFockSolver solver(
        basis, gradients, { { { 0, 0, 0 }, 1 }, { { 1.5117, 0, 0 }, 2 } });

    solver.solveSCF();

    return 0;
}
