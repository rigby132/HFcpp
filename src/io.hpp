/**
 * @file io.hpp
 * @author Deniz Güven (s0394473@uni-frankfurt.de)
 * @brief Defines functions for input and output.
 * @version 1.0
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include "basis.hpp"
#include "scf.hpp"

#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace hf {

// TODO expand?
const std::map<std::string, int> ATOMIC_CHARGES
    = { { "H", 1 }, { "He", 2 }, { "HE", 2 }, { "Li", 3 }, { "LI", 3 }, { "Be", 4 }, { "BE", 4 },
          { "B", 5 }, { "C", 6 }, { "N", 7 }, { "O", 8 }, { "F", 9 } };

/**
 * @brief Constructs a molecule(Nucleus vector) from a .xyz file.
 *
 * @tparam FLOAT Type of floating point to use.
 * @param path The path to the .xyz file.
 * @return A vector of nuclei (position & charge).
 */
template <typename FLOAT = double>
std::vector<Nucleus<FLOAT>> readStructureFromFile(const std::string& path)
{
    std::ifstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    // TODO ERROR HANDLING
    size_t size;
    fileStream >> size;

    // if (size < 1)
    //    throw std::invalid_argument("Number of atoms must be at least 1.");

    std::vector<Nucleus<FLOAT>> structure(size);
    for (size_t i = 0; i < size; i++) {
        std::string atom;

        FLOAT x, y, z;
        int charge;
        fileStream >> atom >> x >> y >> z;

        try {
            charge = ATOMIC_CHARGES.at(atom);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("Element in file is not known.");
        }

        const FLOAT A2au = 1.8897259885789;
        structure[i] = Nucleus<FLOAT> { x * A2au, y * A2au, z * A2au, charge };
    }

    return structure;
}

/**
 * @brief Reads a turbomole basis file and constructs the corresponding basis.
 *
 * @tparam FLOAT The floating point type to use.
 * @param path The path to the file.
 * @param structure A vector of nuclei defining the molecular structure.
 *
 * @return A vector of basisfunctions for each atom.
 */
template <typename FLOAT = double>
std::vector<CGTO<FLOAT>> readBasisFromFile(
    const std::string& path, const std::vector<Nucleus<FLOAT>>& structure)
{
    std::ifstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    // uint: element, pair: [uint: angular momentum, vector: coeff(first) & expo(second) pairs]
    std::map<size_t, std::vector<std::tuple<int, std::vector<FLOAT>, std::vector<FLOAT>>>> basisSet;

    while (true) {
        int element;
        std::string name;
        std::string skip;
        // Get second name and skip the orbital line
        fileStream >> skip >> name >> skip >> skip >> skip >> skip >> skip;

        if (fileStream.eof())
            break;

        try {
            element = ATOMIC_CHARGES.at(name);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("Element in file is not known.");
        }

        unsigned size;
        fileStream >> size;
        for (unsigned i = 0; i < size; i++) {
            int index, momentum, pairs;
            fileStream >> index >> momentum >> pairs;

            std::vector<FLOAT> coeffs;
            std::vector<FLOAT> expos;

            for (int j = 0; j < pairs; j++) {
                FLOAT coeff, expo;
                fileStream >> expo >> coeff;

                coeffs.push_back(coeff);
                expos.push_back(expo);
            }
            basisSet[element].push_back({ momentum, coeffs, expos });
        }
    };

    std::vector<CGTO<FLOAT>> basis;
    for (const auto& atom : structure) {
        std::vector<std::tuple<int, std::vector<FLOAT>, std::vector<FLOAT>>> atomBasis;
        try {
            atomBasis = basisSet.at(atom.charge);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("No basis for this atom available.");
        }

        for (const auto& functions : atomBasis) {
            const int momentum = std::get<0>(functions);

            // Add all permutations of the 3 shape parameters.
            for (int i = 0; i <= momentum; i++)
                for (int j = 0; j <= momentum; j++)
                    for (int k = 0; k <= momentum; k++) {
                        if (i + j + k == momentum) {
                            basis.push_back(CGTO<FLOAT>(atom.x, atom.y, atom.z,
                                std::get<1>(functions), std::get<2>(functions), i, j, k));
                        }
                    }
        }
    }

    return basis;
}

// Legacy format, TODO update or remove.
template <typename FLOAT = double>
void writeOrbitalsVDK(
    const HFSolver& solver, const std::string& path, const FLOAT space, const int points)
{
    std::ofstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    fileStream << "# vtk DataFile Version 4.2\n"
               << "orbital functions\n"
               << "ASCII\n"
               << "DATASET STRUCTURED_POINTS\n";

    FLOAT minX = 0;
    FLOAT minY = 0;
    FLOAT minZ = 0;
    FLOAT maxX = 0;
    FLOAT maxY = 0;
    FLOAT maxZ = 0;

    // x coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.x < minX)
            minX = nuc.x;
        if (nuc.x > maxX)
            maxX = nuc.x;
    }
    // y coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.y < minY)
            minY = nuc.y;
        if (nuc.y > maxY)
            maxY = nuc.y;
    }
    // z coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.z < minZ)
            minZ = nuc.z;
        if (nuc.z > maxZ)
            maxZ = nuc.z;
    }

    // FLOAT points = 50;
    FLOAT spaceX = (maxX - minX + space * 2) / static_cast<FLOAT>(points);
    FLOAT spaceY = (maxY - minY + space * 2) / static_cast<FLOAT>(points);
    FLOAT spaceZ = (maxZ - minZ + space * 2) / static_cast<FLOAT>(points);

    fileStream << "DIMENSIONS " << points << ' ' << points << ' ' << points << '\n'
               << "ORIGIN " << minX - space << ' ' << minY - space << ' ' << minZ - space << '\n'
               << "SPACING " << spaceX << ' ' << spaceY << ' ' << spaceZ << '\n'
               << "POINT_DATA " << static_cast<int>(points * points * points) << '\n';

    // Add data set for each orbital.
    for (int i = 0; i < solver.m_basisSize; i++) {
        fileStream << "SCALARS ro" << std::setfill('0') << std::setw(3) << i << std::setw(0)
                   << " float 1\n"
                   << "LOOKUP_TABLE default\n";

        // Fill in data.
        for (int x = 0; x < points; x++)
            for (int y = 0; y < points; y++)
                for (int z = 0; z < points; z++) {
                    FLOAT ro = solver.orbital(spaceX * x + minX - space, spaceY * y + minY - space,
                        spaceZ * z + minZ - space, i);

                    fileStream << ro * ro << ' ';
                }
        fileStream << '\n';
    }
}

template <typename FLOAT = double>
void writeOrbitalsCUBE(const HFSolver& solver, const std::string& path, const FLOAT space,
    const FLOAT spacing, const bool exportDensity, int numberOfOrbitals)
{
    std::ofstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    if (numberOfOrbitals == 0)
        numberOfOrbitals = solver.m_basisSize;

    // Standard comments in header.
    fileStream << "CPMD CUBE FILE.\n"
               << "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n";

    FLOAT minX = 0;
    FLOAT minY = 0;
    FLOAT minZ = 0;
    FLOAT maxX = 0;
    FLOAT maxY = 0;
    FLOAT maxZ = 0;

    // x coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.x < minX)
            minX = nuc.x;
        if (nuc.x > maxX)
            maxX = nuc.x;
    }
    // y coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.y < minY)
            minY = nuc.y;
        if (nuc.y > maxY)
            maxY = nuc.y;
    }
    // z coordinate.
    for (const auto& nuc : solver.m_nuclei) {
        if (nuc.z < minZ)
            minZ = nuc.z;
        if (nuc.z > maxZ)
            maxZ = nuc.z;
    }

    int pointsX = 1 + (maxX - minX + space * 2) / spacing;
    int pointsY = 1 + (maxY - minY + space * 2) / spacing;
    int pointsZ = 1 + (maxZ - minZ + space * 2) / spacing;

    // Add number of atoms, origin and spacing info.
    fileStream << '-' << solver.m_nuclei.size() << ' ' << minX - space << ' ' << minY - space << ' '
               << minZ - space << '\n'
               << -pointsX << ' ' << spacing << ' ' << 0 << ' ' << 0 << '\n'
               << -pointsY << ' ' << 0 << ' ' << spacing << ' ' << 0 << '\n'
               << -pointsZ << ' ' << 0 << ' ' << 0 << ' ' << spacing << '\n';

    // Add nuclei and their positions.
    for (auto nucleus : solver.m_nuclei)
        fileStream << nucleus.charge << ' ' << 0 << ' ' << nucleus.x << ' ' << nucleus.y << ' '
                   << nucleus.z << '\n';

    // Add Orbital indexing
    fileStream << numberOfOrbitals << '\n';
    for (int i = 0; i < numberOfOrbitals; i++)
        fileStream << i << '\n';

    // Add data set for each orbital.
    if (exportDensity)
        for (int x = 0; x < pointsX; x++)
            for (int y = 0; y < pointsY; y++) {
                for (int z = 0; z < pointsZ; z++)
                    for (int i = 0; i < numberOfOrbitals; i++) {
                        FLOAT ro = solver.orbital(spacing * x + minX - space,
                            spacing * y + minY - space, spacing * z + minZ - space, i);

                        fileStream << ro * ro;

                        if ((z * numberOfOrbitals + i) % 6 == 5) {
                            if (z + 1 < pointsZ or i + 1 < numberOfOrbitals)
                                fileStream << '\n';
                        } else
                            fileStream << ' ';
                    }
                fileStream << '\n';
            }
    else // Export wavefunction.
        for (int x = 0; x < pointsX; x++)
            for (int y = 0; y < pointsY; y++) {
                for (int z = 0; z < pointsZ; z++)
                    for (int i = 0; i < numberOfOrbitals; i++) {
                        FLOAT ro = solver.orbital(spacing * x + minX - space,
                            spacing * y + minY - space, spacing * z + minZ - space, i);

                        fileStream << ro;

                        if ((z * numberOfOrbitals + i) % 6 == 5) {
                            if (z + 1 < pointsZ or i + 1 < numberOfOrbitals)
                                fileStream << '\n';
                        }
                        else
                            fileStream << ' ';
                    }
                fileStream << '\n';
            }
}

}
