#include "poisoning/calc_loss.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace poisoning {

static inline std::pair<long double, long double> mean_var_y(std::size_t n) {
    if (n == 0) throw std::runtime_error("Empty data");
    long double mean_y = (static_cast<long double>(n) + 1.0L) / 2.0L;
    long double var_y  = (static_cast<long double>(n) * n - 1.0L) / 12.0L;
    return {mean_y, var_y};
}

static inline void normalize_u64_to_ld(
    const std::vector<std::uint64_t>& xs,
    long double& offset_ld, long double& scale_ld,
    std::vector<long double>& xnorm
) {
    if (xs.empty()) throw std::runtime_error("Empty data");
    auto [mn_it, mx_it] = std::minmax_element(xs.begin(), xs.end());
    std::uint64_t offset = *mn_it;
    std::uint64_t range  = *mx_it - offset;

    unsigned shift = 0;
    if (range > 0) {
        unsigned lg = 63u - static_cast<unsigned>(__builtin_clzll(range));
        shift = (lg > 40u) ? (lg - 40u) : 0u;
    }
    std::uint64_t scale = (shift == 0) ? 1ULL : (1ULL << shift);

    offset_ld = static_cast<long double>(offset);
    scale_ld  = static_cast<long double>(scale);

    xnorm.resize(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        xnorm[i] = static_cast<long double>(xs[i] - offset) / scale_ld;
    }
}

template <class XContainer>
static inline LinearRegressionStats stats_from_centered_scaled_x(const XContainer& xnorm,
                                                                 long double offset_ld,
                                                                 long double scale_ld)
{
    const std::size_t n = xnorm.size();
    if (n == 0) throw std::runtime_error("Empty data");

    auto [mean_y, var_y] = mean_var_y(n);

    long double mean_xp = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        mean_xp += xnorm[i];
    }
    mean_xp /= static_cast<long double>(n);

    long double sum_xy = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        long double yi = static_cast<long double>(i + 1);
        sum_xy += xnorm[i] * yi;
    }
    long double cov_xp = sum_xy / static_cast<long double>(n) - mean_xp * mean_y;

    long double sum_x2 = 0.0L;
    for (const auto& x : xnorm) {
        sum_x2 += x * x;
    }
    long double var_xp = sum_x2 / static_cast<long double>(n) - mean_xp * mean_xp;

    long double var_x  = var_xp * (scale_ld * scale_ld);
    long double cov_xy = cov_xp * scale_ld;

    long double mean_x = offset_ld + scale_ld * mean_xp;

    if (var_x == 0.0L)
        throw std::runtime_error("Zero variance in x values");

    long double loss = var_y - (cov_xy * cov_xy) / var_x;

    return {mean_x, mean_y, var_x, var_y, cov_xy, loss};
}


double calc_loss(const std::vector<std::uint64_t>& xs_u64) {
    long double offset_ld, scale_ld;
    std::vector<long double> xnorm;
    normalize_u64_to_ld(xs_u64, offset_ld, scale_ld, xnorm);
    auto s = stats_from_centered_scaled_x(xnorm, offset_ld, scale_ld);
    return static_cast<double>(s.loss);
}

double calc_loss(const std::vector<double>& xs) {
    if (xs.empty()) throw std::runtime_error("Empty data");
    std::vector<long double> xld(xs.begin(), xs.end());
    auto s = stats_from_centered_scaled_x(xld, /*offset*/0.0L, /*scale*/1.0L);
    return static_cast<double>(s.loss);
}

LinearRegressionStats compute_stats(const std::vector<std::uint64_t>& xs_u64) {
    long double offset_ld, scale_ld;
    std::vector<long double> xnorm;
    normalize_u64_to_ld(xs_u64, offset_ld, scale_ld, xnorm);
    return stats_from_centered_scaled_x(xnorm, offset_ld, scale_ld);
}

LinearRegressionStats compute_stats(const std::vector<double>& xs) {
    if (xs.empty()) throw std::runtime_error("Empty data");
    std::vector<long double> xld(xs.begin(), xs.end());
    return stats_from_centered_scaled_x(xld, 0.0L, 1.0L);
}

template<>
double calc_loss(const std::vector<std::uint32_t>& xs) {
    std::vector<std::uint64_t> xs_u64(xs.begin(), xs.end());
    return calc_loss(xs_u64);
}

template<>
double calc_loss(const std::vector<std::uint64_t>& xs) {
    return calc_loss(xs);
}

template<>
double calc_loss(const std::vector<double>& xs) {
    return calc_loss(xs);
}

double loss_after_insert(double x_new, size_t pos,
                           size_t n, double sum_x, double sum_x2, double s_xy,
                           double total_sum_x, const std::vector<double>& prefix_sum) {
    size_t n_new = n + 1;
    double sum_x_new = sum_x + x_new;
    double sum_x2_new = sum_x2 + x_new * x_new;
    double mean_x_new = sum_x_new / n_new;
    double var_x_new = sum_x2_new / n_new - mean_x_new * mean_x_new;
    
    double mean_y_new = static_cast<double>(n_new + 1) / static_cast<double>(2);
    double var_y_new = static_cast<double>(n_new * n_new - 1) / static_cast<double>(12);
    
    double sum_after = total_sum_x - (pos > 0 ? prefix_sum[pos - 1] : double(0));
    double s_xy_new = s_xy + x_new * (pos + static_cast<double>(1)) + sum_after;
    double cov_xy_new = s_xy_new / n_new - mean_x_new * mean_y_new;
    
    return var_y_new - cov_xy_new * cov_xy_new / var_x_new;
}

} // namespace poisoning