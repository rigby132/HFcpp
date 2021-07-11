#include "basis.hpp"
#include "boys.hpp"
#include "io.hpp"
#include "scf.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

int main()
{

    auto structure = hf::readStructureFromFile<double>("water.xyz");
    auto basis = hf::readBasisFromFile<double>("6-311G.d2k", structure);

    hf::HFSolver solver(basis, structure, 10);

    solver.solve(0.000001);

    hf::writeOrbitals<float>(solver, "out.vtk", 2, 120);

    return 0;
}
