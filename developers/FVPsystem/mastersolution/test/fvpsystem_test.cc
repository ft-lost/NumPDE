/**
 * @file FVPsystem_test.cc
 * @brief NPDE homework FVPsystem code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../fvpsystem.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace FVPsystem::test {
TEST(FVPsystem, ev1ExpPSystem) {
  // Parameters of the simulation
  double a = -10;
  double b = 10;
  double T = 2;
  unsigned int N = 10;
  unsigned int M = 10;

  // Define the initial condition
  auto u0 = [](double x) -> Eigen::Vector2d {
    Eigen::Vector2d out(2);
    if (x <= 0) {
      out(0) = 1;
      out(1) = 1;
    } else {
      out(0) = 3;
      out(1) = 4;
    }
    return out;
  };

  // Solve the system
  auto mu = FVPsystem::ev1ExpPSystem(a, b, T, N, M, u0);

  // Define the exact solution
  Eigen::MatrixXd mu_exact(2, 10);
  mu_exact << 1.08003, 1.36525, 1.95354, 2.45514, 2.62251, 2.64078, 2.6503,
      2.69424, 2.77155, 2.85881, 1.13887, 1.71332, 3.19006, 4.7415, 5.37246,
      5.46957, 5.42805, 5.26238, 4.96051, 4.60526;

  // Compare exact and computed solution
  ASSERT_LE((mu_exact - mu).norm(), 1e-3);
}

}  // namespace FVPsystem::test
