#pragma once

#include <array>
#include <cstddef>

namespace basis
{

double gto(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha);

double gtoGradient(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha);

template<size_t N>
double cgto(const std::array<double, 3> coord, const std::array<double, 3> center, const std::array<double, N> c, const std::array<double, N> alpha)
{
    double sum = 0;

    for(unsigned int i = 0; i < N; i++)
        sum += gto(coord, center, c[i], alpha[i]);

    return sum;
}

template<size_t N>
double cgtoGradient(const std::array<double, 3> coord, const std::array<double, 3> center, const std::array<double, N> c, const std::array<double, N> alpha)
{
    double sum = 0;

    for(unsigned int i = 0; i < N; i++)
        sum += gtoGradient(coord, center, c[i], alpha[i]);

    return sum;
}

}