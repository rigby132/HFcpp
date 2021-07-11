/**
 * @file basis.hpp
 * @author Deniz Güven (s0394473@uni-frankfurt.de)
 * @brief Implements cgto basis functions.
 * @version 0.1
 * @date 2021-04-01
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <pstl/pstl_config.h>
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
 * @brief Information of the Nucleus: Position[0-2] and charge[3]
 *
 * @tparam T Type of number.
 */
template <typename T = double> struct Nucleus {
    T x, y, z;
    unsigned int charge;
};

/**
 * @brief Factorized GTO function.
 *
 * This is a primitive Gaussian Type Orbital in 1 dimension.
 *
 * @param x The position of evaluation.
 * @param exponent The exponent used.
 * @param cx The center coordinate of the gaussian.
 * @param quantum The quantum number or angular momentum of the gaussian.
 *
 * @return The value of this gaussian type orbital at the specified position.
 */
template <typename FLOAT = double> FLOAT pgto(FLOAT x, FLOAT exponent, FLOAT cx, int quantum)
{
    FLOAT pow = 1;
    for (int i = 0; i < quantum; i++)
        pow *= x;

    return pow * std::exp(-exponent * (x - cx) * (x - cx));
}

template <typename FLOAT = double> class GTO {
public:
    /**
     * @brief In order: Coefficient, exponent for all 3 primitive GTOs.
     */
    const FLOAT c_, a_;

    /**
     * @brief The orbital angular momentum for each primitive GTO.
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
    FLOAT operator()(FLOAT xA, FLOAT yA, FLOAT zA) const
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

    /**
     * @brief Calculates the laplacian of this GTO.
     *
     * This method uses the recurrence relation:
     * GTO(deriv+1, quant) = -2*expo*GTO(deriv, quant+1) + quant*GTO(deriv, quant-1)
     *
     * @param x The x coordinate relative to the center.
     *
     * @return The laplacian at the specified coordinates.
     */
    FLOAT laplace(FLOAT xA, FLOAT yA, FLOAT zA) const
    {
        auto xlap = 4 * a_ * a_ * xA * xA - 2 * a_ * (2 * i_ + 1) + (i_ * i_ - i_) / (xA * xA);
        auto ylap = 4 * a_ * a_ * yA * yA - 2 * a_ * (2 * j_ + 1) + (j_ * j_ - j_) / (yA * yA);
        auto zlap = 4 * a_ * a_ * zA * zA - 2 * a_ * (2 * k_ + 1) + (k_ * k_ - k_) / (zA * zA);

        return (xlap + ylap + zlap) * (*this)(xA, yA, zA);
    }
};

template <typename FLOAT = double> class CGTO {
private:
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
            std::vector<GTO<FLOAT>> gtos; // TODO allocate all at once
            for (size_t f = 0; f < size_; f++)
                gtos.push_back({ coefficients[f], exponents[f], i, j, k });
            return gtos;
        }())
    {
        assert(coefficients.size() == exponents.size());
        assert(i >= 0);
        assert(j >= 0);
        assert(k >= 0);
    }

    /**
     * @brief Creates a CGTO with all of its coefficients divided by norm.
     *
     * @param norm The root of the self overlap of this CGTO.
     * @return The normalized CGTO.
     */
    CGTO normalize(FLOAT norm) const
    {
        std::vector<FLOAT> coeffs;
        std::vector<FLOAT> expos;

        for (const GTO<FLOAT>& gto : gtos_) {
            coeffs.push_back(gto.c_ / norm);
            expos.push_back(gto.a_);
        }

        return CGTO(cx_, cy_, cz_, coeffs, expos, i_, j_, k_);
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
    FLOAT operator()(FLOAT x, FLOAT y, FLOAT z) const
    {
        FLOAT sum = 0;
        for (const GTO<FLOAT>& gto : gtos_)
            sum += gto(x - cx_, y - cy_, z - cz_);

        return sum;
    }

    /**
     * @brief Calculates the laplacian of this CGTO.
     *
     * This method uses the recurrence relation:
     * GTO(deriv+1, quant) = -2*expo*GTO(deriv, quant+1) + quant*GTO(deriv, quant-1)
     *
     * @param x The x coordinate.
     * @param y The y coordinate.
     * @param z The z coordinate.
     *
     * @return The laplacian at the specified coordinates.
     */
    FLOAT laplace(FLOAT x, FLOAT y, FLOAT z) const
    {
        FLOAT sum = 0;

        for (const GTO<FLOAT>& gto : gtos_)
            sum += gto.laplace(x - cx_, y - cy_, z - cz_);

        return sum;
    }
};

template <typename FLOAT = double>
std::array<FLOAT, 3> estimateMean(const CGTO<FLOAT>& fn0, const CGTO<FLOAT>& fn1)
{
    static_assert(std::is_floating_point_v<FLOAT>, "Must use a floating point type.");

    return { 0.5 * (fn0.cx_ + fn1.cx_), 0.5 * (fn0.cy_ + fn1.cy_), 0.5 * (fn0.cz_ + fn1.cz_) };
}

template <typename FLOAT = double>
std::array<FLOAT, 3> estimateDev(const CGTO<FLOAT>& fn0, const CGTO<FLOAT>& fn1)
{
    static_assert(std::is_floating_point_v<FLOAT>, "Must use a floating point type.");

    FLOAT dx = std::abs(fn0.cx_ - fn1.cx_);
    FLOAT dy = std::abs(fn0.cy_ - fn1.cy_);
    FLOAT dz = std::abs(fn0.cz_ - fn1.cz_);

    FLOAT expo0 = 0;
    FLOAT expo1 = 0;

    for (unsigned int i = 0; i < fn0.size_; i++)
        expo0 += std::abs(fn0.gtos_[i].c_ * fn0.gtos_[i].a_);

    for (unsigned int i = 0; i < fn1.size_; i++)
        expo1 += std::abs(fn1.gtos_[i].c_ * fn1.gtos_[i].a_);

    FLOAT k = expo0 + expo1;

    return { std::sqrt(dx + k), std::sqrt(dy + k), std::sqrt(dz + k) };
}
}