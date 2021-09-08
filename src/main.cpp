#include "basis.hpp"
#include "boys.hpp"
#include "io.hpp"
#include "scf.hpp"

#include "CLI11.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>

int main(int argc, const char** argv)
{

    CLI::App app { "HFcpp, a ground state energy solver using the Hartree-Fock method." };

    std::string structureFile;
    app.add_option("-s,--structure", structureFile,
           "Path to a xyz file defining the structure of the molecule.")
        ->required();

    std::string basisFile;
    app.add_option(
           "-b,--basis", basisFile, "Path to a turbomol basis file defining the basis to use.")
        ->required();

    unsigned int occupation;
    app.add_option("-n,--occupation", occupation,
           "First n fully occupied orbitals starting from the lowest one.")
        ->required();

    double tolerance;
    app.add_option(
           "-t,--tolerance", tolerance, "The allowed tolerance to use during the SCF-procedure.")
        ->default_val(0.00001)
        ->check(CLI::PositiveNumber);

    app.add_flag("--vdk", "Output orbital densities as a .vdk file.");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    std::cout << "READING STRUCTURE...\n";
    auto structure = hf::readStructureFromFile<double>(structureFile);

    std::cout << "READING BASIS FUNCTIONS...\n";
    auto basis = hf::readBasisFromFile<double>(basisFile, structure);

    std::cout << "SOLVING...\n";
    hf::HFSolver solver(basis, structure, occupation);
    solver.solve(tolerance);

    if (app.count("--vdk") > 0){
        std::cout << "WRITING OUTPUT...\n";
        hf::writeOrbitals<double>(solver, "out.vdk", 2, 120);
    }
    return 0;
}
