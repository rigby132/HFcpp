#include "basis.hpp"
#include "scf.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "scf\n";

    std::array<double, 3> pos0 { 0, 0, 0 };
    std::array<double, 3> pos1 { 1.5117, 0, 0 };

    std::vector<hf::BasisFunction<3, double>> basis
        = { std::bind(basis::gto, std::placeholders::_1, pos0, 0.3696, 0.41660),
              std::bind(basis::gto, std::placeholders::_1, pos1, 0.5881, 0.77390) };

    std::vector<hf::BasisFunction<3, double>> gradients
        = { std::bind(basis::gtoGradient, std::placeholders::_1, pos0, 0.3696, 0.41660),
              std::bind(basis::gtoGradient, std::placeholders::_1, pos1, 0.5881, 0.77390) };

    hf::HFSolver solver(basis, gradients, { { 0, 0, 0, 1 }, { 1.5117, 0, 0, 2 } }, 2);

    solver.solve(0.00001);

    return 0;
}
