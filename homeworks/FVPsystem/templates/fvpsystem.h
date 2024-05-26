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

  /* Your code goes here! */

  // Return
  return mu;
}
/* SAM_LISTING_END_1 */

}  // namespace FVPsystem

#endif
