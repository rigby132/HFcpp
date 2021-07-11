/**
 * @file boys.hpp
 * @author Deniz Güven
 * @brief Implements the Boys-function.
 * @version 0.1
 * @date 2021-04-15
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <assert.h>
#include <cmath>
#include <math.h>

namespace hf {

/**
 * @brief The boys function of order n.
 *
 * @tparam FLOAT The floating point type to use.
 * @param n The order of the function.
 * @param x The x coordinate to evaluate the function at.
 * @return Value of the boys function.
 */
template <typename FLOAT = double> FLOAT boys(const int n, const FLOAT x)
{
    assert(n >= 0);
    assert(x >= 0);

    if (n == 0 and x == 0)
        return 1;

    constexpr FLOAT pi = 3.141592653589793;
    if (n == 0)
        return std::sqrt(pi / (4 * x)) * std::erf(std::sqrt(x));

    // TODO replace bad approximation with a good one!
    FLOAT f0 = 0;
    for (unsigned int k = 0; k <= 6; k++)
        f0 += std::pow(-x, k) / (std::tgamma(k + 1) * (2 * n + 2 * k + 1));

    int semiFactorial = 1;
    for (int i = 2 * n - 1; i > 0; i -= 2)
        semiFactorial *= i;

    FLOAT f1 = std::sqrt(pi / std::pow(x, 2 * n + 1)) * semiFactorial / std::pow(2, n + 1);

    return std::min(f0, f1);
}

/**
 * @brief The boys function of order n calculated by down/upward recursion.
 *
 * @tparam FLOAT The floating point type to use.
 * @param n The order of the function.
 * @param x The x coordinate to evaluate the function at.
 * @param fn The order of the already evaluated boys function.
 * @param fx The boys function value at x of order boys_n.
 * @return Value of the boys function.
 */
template <typename FLOAT = double> FLOAT boys(const int n, const FLOAT x, int fn, FLOAT fx)
{
    assert(n >= 0);
    assert(x >= 0);
    assert(fn >= 0);
    assert(fx <= 1 / (2 * fn + 1));

    // Downward recursion.
    while (fn > n) {
        fx = (2 * x * fx + std::exp(-x)) / (2 * fn + 1);
        fn--;
    }
    // Upward recursion.
    while (fn < n) {
        assert(x > 0);
        fx = ((2 * fn + 1) * fx - std::exp(-x)) / (2 * x);
        fn++;
    }

    return fx;
}

}