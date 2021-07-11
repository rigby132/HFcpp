/**
 * @file integration.hpp
 * @author Deniz Güven (s0394473@uni-frankfurt.de)
 * @brief Implements monte carlo integration with custom distributions.
 * @version 0.1
 * @date 2021-03-14
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <type_traits>

namespace hf {

/** @brief The point density function for a normal distribution.
 *
 * @param x A point in the distribution.
 * @param mean The mean of the distribution.
 * @param dev The standard deviation of the distribution.
 * @return The point density of the provided point.
 */
template <typename FLOAT = double> FLOAT normalPDF(const FLOAT x, const FLOAT mean, const FLOAT dev)
{
    static_assert(std::is_floating_point_v<FLOAT>, "Must use a floating point type");

    constexpr FLOAT PI = 3.141592653589793238462643;
    constexpr FLOAT sqrt_2PI = std::sqrt(2 * PI);
    const FLOAT factor = 1.0 / (dev * sqrt_2PI);

    FLOAT expo = (x - mean) * (x - mean) / (dev * dev);

    return factor * std::exp(-0.5 * expo);
}

template <typename FLOAT = double, typename FUNC>
FLOAT mc_integrate3(const FUNC& fn, const std::array<FLOAT, 3> mean, const std::array<FLOAT, 3> dev,
    const unsigned long sample_size)
{
    static_assert(std::is_floating_point_v<FLOAT>, "Must use a floating point type");

    std::default_random_engine rnd(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<FLOAT> distX { mean[0], dev[0] };
    std::normal_distribution<FLOAT> distY { mean[1], dev[1] };
    std::normal_distribution<FLOAT> distZ { mean[2], dev[2] };

    FLOAT sum = 0;
    for (unsigned long i = 0; i < sample_size; i++) {
        FLOAT x = distX(rnd);
        FLOAT y = distY(rnd);
        FLOAT z = distZ(rnd);

        FLOAT pdf = normalPDF(x, mean[0], dev[0]) * normalPDF(y, mean[1], dev[1])
            * normalPDF(z, mean[2], dev[2]);
        sum += fn(x, y, z) / pdf;
    }

    return sum / sample_size;
}

template <typename FLOAT = double, typename FUNC>
FLOAT mc_integrate6(const FUNC& fn, const std::array<FLOAT, 6> mean, const std::array<FLOAT, 6> dev,
    const unsigned long sample_size)
{
    static_assert(std::is_floating_point_v<FLOAT>, "Must use a floating point type!");

    std::default_random_engine rnd(std::chrono::system_clock::now().time_since_epoch().count());
    std::array<std::normal_distribution<FLOAT>, 6> dists;
    for (unsigned int i = 0; i < 6; i++)
        dists[i] = std::normal_distribution<FLOAT> { mean[i], dev[i] };

    FLOAT sum = 0;
    for (unsigned long i = 0; i < sample_size; i++) {
        std::array<FLOAT, 6> x;
        FLOAT pdf = 1;
        for (unsigned int j = 0; j < 6; j++) {
            x[j] = dists[j](rnd);
            pdf *= normalPDF(x[j], mean[j], dev[j]);
        }
        sum += fn(x[0], x[1], x[2], x[3], x[4], x[5]) / pdf;
    }

    return sum / sample_size;
}
}