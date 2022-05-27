/**
 * @file basis.hpp
 * @author Deniz Güven (s0394473@uni-frankfurt.de)
 * @brief Implements cgto basis functions.
 * @version 1.0
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

namespace hf {

enum ANGULAR_MOMENTUM {
    S = 0,
    P = 1,
    D = 2,
    F = 3,
    G = 4, // Higher values possible but no names are assigned to these.
};

/**
 * @brief Information of the Nucleus: Position and charge.
 *
 * @tparam T Type of number.
 */
template <typename T = double> struct Nucleus {
    T x, y, z;
    int charge;
};

/**
 * @brief Gaussian Type Orbital, represents a single Orbital function.
 *
 * @tparam FLOAT The floating point type to use.
 */
template <typename FLOAT = double> class GTO {
public:
    /** @brief In order: Coefficient, exponent for all 3 primitive GTOs.
     */
    const FLOAT c_, a_;

    /** @brief The orbital angular momentum for each primitive GTO.
     */
    const int i_, j_, k_;

public:
    /**
     * @brief Evaluates the GTO.
     *
     * @param xA The x coordinate relative to the center.
     * @param yA The y coordinate relative to the center.
     * @param zA The z coordinate relative to the center.
     *
     * @return The GTO value at the specified point.
     */
    inline FLOAT operator()(FLOAT xA, FLOAT yA, FLOAT zA) const
    {
        FLOAT pow = 1;
        for (int i = 0; i < i_; i++)
            pow *= xA;

        for (int j = 0; j < j_; j++)
            pow *= yA;

        for (int k = 0; k < k_; k++)
            pow *= zA;

        return c_ * pow * std::exp(-a_ * (xA * xA + yA * yA + zA * zA));
    }
};

template <typename FLOAT = double> class CGTO {
public:
    /** @brief The coordinates at which this Orbital is located at.
     */
    const FLOAT cx_, cy_, cz_;
    const int i_, j_, k_;
    const size_t size_;

    /** @brief A list of GTOs for each dimension.
     */
    const std::vector<GTO<FLOAT>> gtos_;

public:
    CGTO(FLOAT centerX, FLOAT centerY, FLOAT centerZ, std::vector<FLOAT> coefficients,
        std::vector<FLOAT> exponents, int i, int j, int k)
        : cx_(centerX)
        , cy_(centerY)
        , cz_(centerZ)
        , i_(i)
        , j_(j)
        , k_(k)
        , size_(coefficients.size())
        , gtos_([=] {
            std::vector<GTO<FLOAT>> gtos;
            gtos.reserve(size_);
            for (size_t f = 0; f < size_; f++)
                gtos.push_back({ coefficients[f], exponents[f], i, j, k });
            return gtos;
        }())
    {
        assert(coefficients.size() == exponents.size());
        static_assert(std::is_floating_point_v<FLOAT>, "CGTO type must use a floating point type!");
    }

    /**
     * @brief Evaluates the CGTO.
     *
     * @param x The x coordinate.
     * @param y The y coordinate.
     * @param z The z coordinate.
     *
     * @return The CGTO value at the specified point.
     */
    inline FLOAT operator()(FLOAT x, FLOAT y, FLOAT z) const
    {
        FLOAT sum = 0;
        for (const GTO<FLOAT>& gto : gtos_)
            sum += gto(x - cx_, y - cy_, z - cz_);

        return sum;
    }
};

}