/**
 * @file fvpsystem.h
 * @brief NPDE homework FVPsystem code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef FVPsystem_H_
#define FVPsystem_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <Eigen/Core>
#include <iostream>

namespace FVPsystem {

/* SAM_LISTING_BEGIN_1 */
template <typename u0Functor,
          typename RECORDER = std::function<void(const Eigen::MatrixXd &)>>
Eigen::MatrixXd ev1ExpPSystem(
    double a, double b, double T, unsigned int N, unsigned int M,
    u0Functor &&u0,
    RECORDER recorder = [](const Eigen::MatrixXd &) -> void {}) {
  //  Determine constants of scheme
  double dt = T / M;
  double h = (b - a) / N;
  Eigen::MatrixXd mu(2, N);  // return vector

  // Set initial conditions
  for (unsigned j = 0; j < N; ++j) {
    mu.col(j) = u0(a + h / 2. + j * h);
  }

  // Define the function p, note that p' = p for this special case
  auto p = [](double u) -> double { return -std::exp(u); };

  // Define the continuous flux function
  auto F = [p](Eigen::Vector2d v) -> Eigen::Vector2d {
    Eigen::Vector2d out;
    out(0) = -v(1);
    out(1) = p(v(0));
    return out;
  };

  // Define the HLLE flux
  auto numflux = [p, F](Eigen::MatrixXd v,
                            Eigen::MatrixXd w) -> Eigen::VectorXd {
    Eigen::VectorXd out(2);
    double smin = std::min(-std::sqrt(-p(v(0))),
                           -std::sqrt(-(p(w(0)) - p(v(0))) / (w(0) - v(0))));
    double smax = std::max(std::sqrt(-p(w(0))),
                           std::sqrt(-(p(w(0)) - p(v(0))) / (w(0) - v(0))));
    Eigen::Vector2d ustar;
    ustar = 1. / (smin - smax) * (F(w) - F(v) - smax * w + smin * v);
    if (smin > 0)
      return F(v);
    else if (smin < 0 && smax > 0)
      return F(ustar);
    else if (smax < 0)
      return F(w);
    else
      return F(w);
  };

  // Solve the ODE
  recorder(mu);
  for (double t = 0; t < T - dt / 2.; t += dt) {
    Eigen::MatrixXd fd(2, mu.cols()); // Flux difference

    // constant continuation of data
    fd.col(0) = numflux(mu.col(0), mu.col(1)) - numflux(mu.col(0), mu.col(0));
    // Fluxes on the interior
    for (unsigned j = 1; j < N - 1; ++j) {
      fd.col(j) = numflux(mu.col(j), mu.col(j + 1)) -
                  numflux(mu.col(j - 1), mu.col(j));  // see \eqref{eq:2pcf}
    }
    // constant continuation of data!
    fd.col(N - 1) = numflux(mu.col(N - 1), mu.col(N - 1)) -
                    numflux(mu.col(N - 2), mu.col(N - 1));

    mu -= dt / h * fd;
    recorder(mu);
  }

  // Return
  return mu;
}
/* SAM_LISTING_END_1 */

}  // namespace FVPsystem

#endif
