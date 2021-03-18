#pragma once

#include <array>

namespace basis
{
double gto(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha);

double gtoGradient(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha);

}