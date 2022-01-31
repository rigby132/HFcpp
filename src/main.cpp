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

    double buffer;
    app.add_option(
           "--buffer", buffer, "The extra space around the nuclei to include in the outputs.")
        ->default_val(2.0)
        ->check(CLI::PositiveNumber);

    double spacing;
    app.add_option(
           "--space", spacing, "The size of each volume element in the output file.")
        ->default_val(0.1)
        ->check(CLI::PositiveNumber);

    app.add_flag("--vdk", "Output orbital densities as a .vdk file.");

    app.add_flag("--cube", "Output orbital data (wavefunction unless specified otherwise) as a .cube file.");

    app.add_flag("--density", "Output orbital data as densities.");

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
        std::cout << "WRITING VDK OUTPUT...\n";
        hf::writeOrbitalsVDK<double>(solver, "out.vdk", buffer, 120);
    }

    if (app.count("--cube") > 0){
        std::cout << "WRITING CUBE OUTPUT...\n";
        if(app.count("--density") > 0)
            hf::writeOrbitalsCUBE<double>(solver, "density.cube", buffer, spacing, true);
        else
            hf::writeOrbitalsCUBE<double>(solver, "wave.cube", buffer, spacing, false);
    }

    return 0;
}
