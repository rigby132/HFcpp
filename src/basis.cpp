#include "basis.hpp"

#include <cmath>

double basis::gto(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha)
{
    double distance_square = (coord[0] - center[0]) * (coord[0] - center[0])
        + (coord[1] - center[1]) * (coord[1] - center[1]) + (coord[2] - center[2]) * (coord[2] - center[2]);
    return c * std::exp(-alpha * distance_square);
}

double basis::gtoGradient(const std::array<double, 3> coord, const std::array<double, 3> center, const double c, const double alpha)
{
    double x = coord[0], y = coord[1], z = coord[2], x0 = center[0], y0 = center[1], z0 = center[2];

    double distance_square = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
    double xterm = 2 * alpha * x * x - 4 * x0 * alpha * x + 2 * x0 * x0 * alpha;
    double yterm = 2 * alpha * y * y - 4 * y0 * alpha * y + 2 * y0 * y0 * alpha;
    double zterm = 2 * alpha * z * z - 4 * z0 * alpha * z + 2 * z0 * z0 * alpha;
    return 2 * alpha * c * (xterm + yterm + zterm - 3) * std::exp(-alpha * distance_square);
}


