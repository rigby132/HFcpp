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

/**
 * @brief Functor with a overload for generating points and for its pdf.
 *
 * Multivariate uncorrelated normal distribution.
 *
 * @tparam N The amount of normal distributions to contain.
 * @tparam T The precision with which to operate.
 */
template <size_t N, typename T = double> class NormalDistribution {
private:
    std::default_random_engine m_rnd;
    std::array<std::normal_distribution<T>, N> m_dist;
    const std::array<T, N> m_mean;
    const std::array<T, N> m_dev;

public:
    NormalDistribution(const std::array<T, N>& mean, const std::array<T, N>& dev)
        : m_rnd(std::chrono::system_clock::now().time_since_epoch().count())
        , m_mean(mean)
        , m_dev(dev)
    {
        for (size_t i = 0; i < N; i++)
            m_dist[i] = std::normal_distribution<T>(mean[i], dev[i]);
    }

    /** @brief Generates a random point according to the distribution.
     *
     * @return A random point in the distribution.
     */
    std::array<T, N> operator()()
    {
        std::array<T, N> point;
        for (size_t i = 0; i < N; i++)
            point[i] = m_dist[i](m_rnd);

        return point;
    }

    /** @brief The point density function for this distribution.
     *
     * @param point A point in the distribution.
     * @return The point density of the provided point.
     */
    T operator()(const std::array<T, N>& point)
    {
        T det_dev = 1;
        for (auto d : m_dev)
            det_dev *= d;

        constexpr T PI = 3.141592653589793238462643;
        const T sqrt_2PI_N = std::sqrt(std::pow(2 * PI, N));
        const T factor = 1.0 / (det_dev * sqrt_2PI_N);

        T expo = 0.0;
        for (size_t i = 0; i < N; i++)
            expo += (point[i] - m_mean[i]) * (point[i] - m_mean[i]) / (m_dev[i] * m_dev[i]);

        return factor * std::exp(-0.5 * expo);
    };
};

/** @brief Integrates a multidimensional function with the monte carlo method and a given
 * distribution.
 *
 * @tparam N The dimension of the provided function.
 * @tparam T The floatig point type to use.
 * @tparam FUNC
 * @tparam DIST Random point generator functor type with 2 overloads: std::array<T, N>() and
 * T(std::array<T, N>).
 * @param fn The function to integrate over, must accept an std::array<T, N> as input.
 * @param dist A distribution with 2 overloads: () and (std::array<T, N>), The  generates a
 * random point and the second is the corresponding pdf.
 * @param sample_size Number of samples to use for the integration.
 * @return The integration value of fn over the distribution.
 */
template <size_t N = 3, typename T = double, typename FUNC, typename DIST>
T mc_integrate(const FUNC& fn, DIST& dist, const unsigned long sample_size)
{
    static_assert(
        std::is_floating_point<T>::value, "Must use a floating point type to integrate with!");

    T sum = 0;

    for (unsigned long i = 0; i < sample_size; i++) {
        const std::array<T, N> point = dist(); // Generate random value;

        sum += fn(point) / dist(point);
    }

    return sum / sample_size;
}

}
