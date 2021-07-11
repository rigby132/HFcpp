/**
 * @file io.hpp
 * @author Deniz Güven (s0394473@uni-frankfurt.de)
 * @brief Defines functions for input and output.
 * @version 0.1
 * @date 2021-07-11
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#include "basis.hpp"
#include "scf.hpp"

#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace hf {

const std::map<std::string, unsigned int> ATOMIC_CHARGES
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
    std::cout << "READING .xyz data: " << path << '\n';
    std::ifstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    int size;
    fileStream >> size;

    if (size < 1)
        throw std::invalid_argument("Number of atoms must be at least 1.");

    std::cout << "No atoms: " << size << '\n';

    std::vector<Nucleus<FLOAT>> structure(size);
    for (unsigned int i = 0; i < static_cast<unsigned int>(size); i++) {
        std::string atom;

        FLOAT x, y, z;
        unsigned int charge;
        fileStream >> atom >> x >> y >> z;
        std::cout << "Reading atom: " << atom << ' ' << x << ' ' << y << ' ' << z << '\n';

        try {
            charge = ATOMIC_CHARGES.at(atom);
        } catch (const std::out_of_range& e) {
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
    std::cout << "READING basis data: " << path << '\n';
    std::ifstream fileStream(path);

    if (!fileStream.is_open())
        throw std::invalid_argument("Could not open file");

    // uint: element, pair: [uint: angular momentum, vector: coeff(first) & expo(second) pairs]
    std::map<unsigned int,
        std::vector<std::tuple<unsigned int, std::vector<FLOAT>, std::vector<FLOAT>>>>
        basisSet;

    while (true) {
        unsigned int element;
        std::string name;
        std::string skip;
        // Get second name and skip the orbital line
        fileStream >> skip >> name >> skip >> skip >> skip >> skip >> skip;

        if (fileStream.eof())
            break;

        // std::cout << "Element: " << name << '\n';
        try {
            element = ATOMIC_CHARGES.at(name);
        } catch (const std::out_of_range& e) {
            throw std::invalid_argument("Element in file is not known.");
        }

        unsigned int size;
        fileStream >> size;
        for (unsigned int i = 0; i < size; i++) {
            unsigned int index, momentum, pairs;
            fileStream >> index >> momentum >> pairs;
            // std::cout << "Data: " << index << ' ' << momentum << ' ' << pairs << '\n';

            std::vector<FLOAT> coeffs;
            std::vector<FLOAT> expos;

            for (unsigned int j = 0; j < pairs; j++) {
                FLOAT coeff, expo;
                fileStream >> expo >> coeff;
                // std::cout << "c&e: " << coeff << ' ' << expo << '\n';
                coeffs.push_back(coeff);
                expos.push_back(expo);
            }
            basisSet[element].push_back({ momentum, coeffs, expos });
        }
    };

    std::vector<CGTO<FLOAT>> basis;
    for (const auto& atom : structure) {
        std::cout << "ATOM: " << atom.charge << '\n';
        std::vector<std::tuple<unsigned int, std::vector<FLOAT>, std::vector<FLOAT>>> atomBasis;
        try {
            atomBasis = basisSet.at(atom.charge);
        } catch (const std::out_of_range& e) {
            throw std::invalid_argument("No basis for this atom available.");
        }

        for (const auto& functions : atomBasis) {
            const int momentum = std::get<0>(functions);

            std::cout << "Momentum: " << momentum << '\n';

            // Add all permutations of the 3 shape parameters.
            for (int i = 0; i <= momentum; i++)
                for (int j = 0; j <= momentum; j++)
                    for (int k = 0; k <= momentum; k++) {
                        if (i + j + k == momentum) {
                            std::cout << "i j k: " << i << ' ' << j << ' ' << k << '\n';
                            basis.push_back(CGTO<FLOAT>(atom.x, atom.y, atom.z,
                                std::get<1>(functions), std::get<2>(functions), i, j, k));
                        }
                    }
        }
    }

    return basis;
}

template <typename FLOAT = double>
void writeOrbitals(
    const HFSolver& solver, const std::string& path, const FLOAT space, const unsigned int points)
{
    std::cout << "WRITING orbital data: " << path << '\n';
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
    for (unsigned int i = 0; i < solver.m_basisSize; i++) {
        fileStream << "SCALARS ro" << std::setfill('0') << std::setw(3) << i << std::setw(0)
                   << " float 1\n"
                   << "LOOKUP_TABLE default\n";

        // Fill in data.
        for (unsigned int x = 0; x < points; x++)
            for (unsigned int y = 0; y < points; y++)
                for (unsigned int z = 0; z < points; z++) {
                    FLOAT ro = solver.orbital(spaceX * x + minX - space, spaceY * y + minY - space,
                        spaceZ * z + minZ - space, i);

                    fileStream << ro * ro << ' ';
                }
        fileStream << '\n';
    }
}

template <typename FLOAT = double>
void writeOrbitalsContour(const HFSolver& solver, const std::string& path, const FLOAT space,
    const unsigned int points, const FLOAT minValue)
{
    std::cout << "WRITING orbital data: " << path << '\n';
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
    for (unsigned int i = 0; i < solver.m_basisSize; i++) {
        fileStream << "SCALARS ro" << std::setfill('0') << std::setw(3) << i << std::setw(0)
                   << " float 1\n"
                   << "LOOKUP_TABLE default\n";

        // Fill in data.
        for (unsigned int x = 0; x < points; x++)
            for (unsigned int y = 0; y < points; y++)
                for (unsigned int z = 0; z < points; z++) {
                    FLOAT ro = solver.orbital(spaceX * x + minX - space, spaceY * y + minY - space,
                        spaceZ * z + minZ - space, i);

                    if (ro * ro >= minValue)
                        fileStream << 1.0f << ' ';
                    else
                        fileStream << 0.0f << ' ';
                }
        fileStream << '\n';
    }
}

}