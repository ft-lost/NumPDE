/**
 * @file FVMIsentropicEuler_test.cc
 * @brief NPDE homework FVMIsentropicEuler code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../fvmisentropiceuler.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

/* Test in the google testing framework

  The following assertions are available, syntax
  EXPECT_XX( ....) << [anything that can be givne to std::cerr]

  EXPECT_EQ(val1, val2)
  EXPECT_NEAR(val1, val2, abs_error) -> should be used for numerical results!
  EXPECT_NE(val1, val2)
  EXPECT_TRUE(condition)
  EXPECT_FALSE(condition)
  EXPECT_GE(val1, val2)
  EXPECT_LE(val1, val2)
  EXPECT_GT(val1, val2)
  EXPECT_LT(val1, val2)
  EXPECT_STREQ(str1,str2)
  EXPECT_STRNE(str1,str2)
  EXPECT_STRCASEEQ(str1,str2)
  EXPECT_STRCASENE(str1,str2)

  "EXPECT" can be replaced with "ASSERT" when you want to program to terminate,
 if the assertion is violated.
 */

namespace FVMIsentropicEuler::test {

TEST(FVMIsentropicEuler, numfluxHLLEIseEul0) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., 3.;
  v << 1., 3.;
  exact_flux << 3., 9.5;
  std::function<double(double)> p = [](double x) -> double {
    return 1. / 2. * x * x;
  };
  std::function<double(double)> pd = [](double x) -> double { return x; };
  Eigen::Vector2d res = numfluxHLLEIseEul(p, pd, u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLEIseEul1) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., 3.;
  v << .5, 4.;
  exact_flux << 3., 9.5;
  std::function<double(double)> p = [](double x) -> double {
    return 1. / 2. * x * x;
  };
  std::function<double(double)> pd = [](double x) -> double { return x; };
  Eigen::Vector2d res = numfluxHLLEIseEul(p, pd, u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLEIseEul2) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., -4.;
  v << .5, -3.;
  exact_flux << -3., 18. + 1. / 8.;
  std::function<double(double)> p = [](double x) -> double {
    return 1. / 2. * x * x;
  };
  std::function<double(double)> pd = [](double x) -> double { return x; };
  Eigen::Vector2d res = numfluxHLLEIseEul(p, pd, u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLEIseEul3) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., -4.;
  v << .5, 3.;
  exact_flux << -0.128442, 0.149371;
  std::function<double(double)> p = [](double x) -> double {
    return 1. / 2. * x * x;
  };
  std::function<double(double)> pd = [](double x) -> double { return x; };
  Eigen::Vector2d res = numfluxHLLEIseEul(p, pd, u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-5);
}

TEST(FVMShallowWater, slopelimfluxdiff_sys1) {
  Eigen::MatrixXd mu(2, 4), exact_fd(2, 4);
  mu << 1., 3., 1., .5, 3., 1., 3., 4.;
  exact_fd << -1., -0.25, 2., 0.25, 2., 0.140625, -2.4375, -0.078125;
  std::function<Eigen::Vector2d(Eigen::Vector2d, Eigen::Vector2d,
                                Eigen::Vector2d)>
      slopes = [](Eigen::Vector2d u1, Eigen::Vector2d u2,
                  Eigen::Vector2d u3) -> Eigen::Vector2d {
    Eigen::Vector2d out;
    for (int j = 0; j < 2; ++j) {
      if (u1(j) > u2(j) && u2(j) > u3(j))
        out(j) = std::max(u2(j) - u1(j), u3(j) - u2(j));
      else if (u1(j) < u2(j) && u2(j) < u3(j))
        out(j) = std::min(u2(j) - u1(j), u3(j) - u2(j));
      else
        out(j) = 0;
    }
    return out;
  };
  std::function<Eigen::Vector2d(Eigen::Vector2d, Eigen::Vector2d)> F =
      [](Eigen::Vector2d u, Eigen::Vector2d v) -> Eigen::Vector2d {
    Eigen::Vector2d out;
    out << (u(1) + v(1)) / 2., (u(0) * u(0) + v(0) * v(0)) / 4.;
    return out;
  };

  Eigen::MatrixXd res = slopelimfluxdiff_sys(mu, F, slopes);

  ASSERT_NEAR((res - exact_fd).norm(), 0., 1e-8);
}

TEST(FVMShallowWater, musclIseEul) {
  double a = -2.;
  double b = 2.;
  double g = 1;
  double T = 1.;
  double pastL2error = 10000;
  for (int N = 256; N < 1000; N *= 2) {
    double h = (b - a) / N;
    Eigen::MatrixXd mu0_mat(2, N);
    for (int i = 0; i < N; ++i) {
      mu0_mat(0, i) = 1.;
      if (i < N / 2)
        mu0_mat(1, i) = .5;
      else
        mu0_mat(1, i) = -0.5;
    }
    std::function<double(double)> p = [g](double x) -> double {
      return 1. / 2. * g * x * x;
    };
    std::function<double(double)> dp = [g](double x) -> double {
      return g * x;
    };
    Eigen::Matrix res = musclIseEul(a, b, T, mu0_mat, p, dp);

    // Compute the error in the smooth parts of the domain
    double L2err = 0.;
    for (int i = 0; i < N / 4; ++i) {
      L2err += (res(0, i) - 1) * (res(0, i) - 1) * h +
               (res(1, i) - 0.5) * (res(1, i) - 0.5) * h;
    }
    for (int i = (3 * N) / 8; i < (5 * N) / 8; ++i) {
      L2err += (res(0, i) - 1.551389963353) * (res(0, i) - 1.551389963353) * h +
               (res(1, i) - 0.000000134979) * (res(1, i) - 0.000000134979) * h;
    }
    for (int i = (3 * N) / 4; i < N; ++i) {
      L2err += (res(0, i) - 1) * (res(0, i) - 1) * h +
               (res(1, i) + 0.5) * (res(1, i) + 0.5) * h;
    }
    L2err = std::sqrt(L2err);
    ASSERT_GE(std::log(pastL2error / L2err) / std::log(2.), 2.);
    pastL2error = L2err;
  }
}

}  // namespace FVMIsentropicEuler::test
