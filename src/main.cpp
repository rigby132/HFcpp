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


    // USE 6-32G basis
    auto H1 = std::bind(basis::cgto<2>, std::placeholders::_1, pos0,
        std::array<double, 2> { 0.1562849787E+00, 0.9046908767E+00 },
        std::array<double, 2> { 0.5447178000E+01, 0.8245472400E+00 });
    auto H2 = std::bind(basis::gto, std::placeholders::_1, pos0, 1, 0.1831915800E+00);

    auto He1 = std::bind(basis::cgto<2>, std::placeholders::_1, pos1,
        std::array<double, 2> { 0.1752298718E+00, 0.8934823465E+00 },
        std::array<double, 2> { 0.1362670000E+02, 0.1999350000E+01 });
    auto He2 = std::bind(basis::gto, std::placeholders::_1, pos1, 1, 0.3829930000E+00);

    auto H1_g = std::bind(basis::cgtoGradient<2>, std::placeholders::_1, pos0,
        std::array<double, 2> { 0.1562849787E+00, 0.9046908767E+00 },
        std::array<double, 2> { 0.5447178000E+01, 0.8245472400E+00 });
    auto H2_g = std::bind(basis::gtoGradient, std::placeholders::_1, pos0, 1, 0.1831915800E+00);

    auto He1_g = std::bind(basis::cgtoGradient<2>, std::placeholders::_1, pos1,
        std::array<double, 2> { 0.1752298718E+00, 0.8934823465E+00 },
        std::array<double, 2> { 0.1362670000E+02, 0.1999350000E+01 });
    auto He2_g = std::bind(basis::gtoGradient, std::placeholders::_1, pos1, 1, 0.3829930000E+00);

    std::vector<hf::BasisFunction<3, double>> basis = { H1, H2, He1, He2 };

    std::vector<hf::BasisFunction<3, double>> gradients = { H1_g, H2_g, He1_g, He2_g };

/*
    std::vector<hf::BasisFunction<3, double>> basis
        = { std::bind(basis::gto, std::placeholders::_1, pos0, 0.3696, 0.41660),
              std::bind(basis::gto, std::placeholders::_1, pos1, 0.5881, 0.77390) };

    std::vector<hf::BasisFunction<3, double>> gradients
        = { std::bind(basis::gtoGradient, std::placeholders::_1, pos0, 0.3696, 0.41660),
              std::bind(basis::gtoGradient, std::placeholders::_1, pos1, 0.5881, 0.77390) };

*/
    hf::HFSolver solver(basis, gradients, { { 0, 0, 0, 1 }, { 1.5117, 0, 0, 2 } }, 2);

    solver.solve(0.000001);

    return 0;
}
