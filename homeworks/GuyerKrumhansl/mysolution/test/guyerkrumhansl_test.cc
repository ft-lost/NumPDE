/**
 * @file guyerKrumhansl_test.cc
 * @brief NPDE homework XXX code
 * @author R. Hiptmair
 * @date July 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../guyerkrumhansl.h"

#include <gtest/gtest.h>

namespace GuyerKrumhansl::test {
TEST(GuyerKrumhansl, generateMOLMatrices_Mmatrix) {
  // Initialize variables for generating MOL matrices
  unsigned int M = 4;
  double rho = 7;
  double sigma = 0.5;
  double mu = 3.5;
  double kappa = 11.;

  // Compute MOL matrices using user's function
  auto res = GuyerKrumhansl::generateMOLMatrices(M, rho, sigma, mu, kappa);
  auto Mtriplets = res.first;
  auto Atriplets = res.second;

  // Correct values for the M-matrix
  std::vector<Eigen::Vector3d> Msolution;
  Msolution.emplace_back(Eigen::Vector3d(0, 0, 1.75));
  Msolution.emplace_back(Eigen::Vector3d(1, 1, 1.75));
  Msolution.emplace_back(Eigen::Vector3d(2, 2, 1.75));
  Msolution.emplace_back(Eigen::Vector3d(3, 3, 1.75));
  Msolution.emplace_back(Eigen::Vector3d(4, 4, 0.0833333));
  Msolution.emplace_back(Eigen::Vector3d(5, 4, 0.0208333));
  Msolution.emplace_back(Eigen::Vector3d(5, 5, 0.0833333));
  Msolution.emplace_back(Eigen::Vector3d(4, 5, 0.0208333));
  Msolution.emplace_back(Eigen::Vector3d(6, 5, 0.0208333));
  Msolution.emplace_back(Eigen::Vector3d(6, 6, 0.0833333));
  Msolution.emplace_back(Eigen::Vector3d(5, 6, 0.0208333));

  // Check if the output of the user corresponds to the correct output
  EXPECT_TRUE(Msolution.size() == Mtriplets.size());
  for (int i = 0; i < Mtriplets.size(); ++i) {
    bool found = false;
    for (int j = 0; j < Mtriplets.size(); ++j) {
      if (std::abs(Msolution.at(i)[0] - Mtriplets.at(i).col()) < 0.1 &&
          std::abs(Msolution.at(i)[1] - Mtriplets.at(i).row()) < 0.1) {
        found = true;
        EXPECT_LT(std::abs(Msolution.at(i)[2] - Mtriplets.at(i).value()), 1e-5);
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(GuyerKrumhansl, generateMOLMatrices_Amatrix) {
  // Initialize variables for generating MOL matrices
  unsigned int M = 4;
  double rho = 7;
  double sigma = 0.5;
  double mu = 3.5;
  double kappa = 11.;

  // Compute MOL matrices using user's function
  auto res = GuyerKrumhansl::generateMOLMatrices(M, rho, sigma, mu, kappa);
  auto Atriplets = res.second;

  // Correct values for the A-matrix
  std::vector<Eigen::Vector3d> Asolution;
  Asolution.emplace_back(Eigen::Vector3d(4, 0, -1));
  Asolution.emplace_back(Eigen::Vector3d(0, 4, 11));
  Asolution.emplace_back(Eigen::Vector3d(5, 1, -1));
  Asolution.emplace_back(Eigen::Vector3d(4, 1, 1));
  Asolution.emplace_back(Eigen::Vector3d(1, 5, 11));
  Asolution.emplace_back(Eigen::Vector3d(1, 4, -11));
  Asolution.emplace_back(Eigen::Vector3d(6, 2, -1));
  Asolution.emplace_back(Eigen::Vector3d(5, 2, 1));
  Asolution.emplace_back(Eigen::Vector3d(2, 6, 11));
  Asolution.emplace_back(Eigen::Vector3d(2, 5, -11));
  Asolution.emplace_back(Eigen::Vector3d(6, 3, 1));
  Asolution.emplace_back(Eigen::Vector3d(3, 6, -11));
  Asolution.emplace_back(Eigen::Vector3d(4, 4, -28.1667));
  Asolution.emplace_back(Eigen::Vector3d(5, 4, 13.9583));
  Asolution.emplace_back(Eigen::Vector3d(5, 5, -28.1667));
  Asolution.emplace_back(Eigen::Vector3d(4, 5, 13.9583));
  Asolution.emplace_back(Eigen::Vector3d(6, 5, 13.9583));
  Asolution.emplace_back(Eigen::Vector3d(6, 6, -28.1667));
  Asolution.emplace_back(Eigen::Vector3d(5, 6, 13.9583));

  // Check if the output of the user corresponds to the correct output
  EXPECT_TRUE(Asolution.size() == Atriplets.size());
  for (int i = 0; i < Atriplets.size(); ++i) {
    bool found = false;
    for (int j = 0; j < Atriplets.size(); ++j) {
      if (std::abs(Asolution.at(i)[0] - Atriplets.at(i).col()) < 0.1 &&
          std::abs(Asolution.at(i)[1] - Atriplets.at(i).row()) < 0.1) {
        found = true;
        EXPECT_LT(std::abs(Asolution.at(i)[2] - Atriplets.at(i).value()), 1e-4);
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(GuyerKrumhansl, timestepping_GKHeat) {
  // Initialize variables for generating mu and zeta
  double T = 1.;
  double L = 1.;
  unsigned int M = 5;
  Eigen::VectorXd mu0(M);
  mu0 << 1., 1., 1., 2., 1.;
  Eigen::VectorXd zeta0(M - 1);
  zeta0 << 1., 3., 1., -2.;
  double rho = 3.;
  double sigma = 5.;
  double mu = 2.;
  double kappa = 3. / 2.;

  // Correct values for the mu and zeta
  Eigen::VectorXd mu_sol(M);
  mu_sol << 0.758846, 0.893403, 1.17565, 2.06852, 1.10358;
  Eigen::VectorXd zeta_sol(M - 1);
  zeta_sol << 0.144693, 0.20865, 0.103263, 0.0621498;

  // Compute mu and zeta using user's function
  auto result = GuyerKrumhansl::timestepping_GKHeat(T, L, M, mu0, zeta0, rho,
                                                    sigma, mu, kappa);
  auto mu_vec = result.first;
  auto zeta_vec = result.second;

  // Check if the output of the user corresponds to the correct output
  for (int i = 0; i < zeta_vec.size(); ++i) {
    EXPECT_LT(std::abs(mu_sol[i] - mu_vec[i]), 1e-5);
    EXPECT_LT(std::abs(zeta_sol[i] - zeta_vec[i]), 1e-5);
  }
}

TEST(GuyerKrumhansl, track_GKHeatEnergy) {
  // Initialize variables for generating energies
  double T = 1.;
  const int L = 5;
  unsigned int M = 5;
  Eigen::VectorXd mu0(M);
  mu0 << 1., 1., 1., 2., 1.;
  Eigen::VectorXd zeta0(M - 1);
  zeta0 << 1., 3., 1., -2.;
  double rho = 3.;
  double sigma = 5.;
  double mu = 2.;
  double kappa = 3. / 2.;

  // Correct values for the energies
  Eigen::VectorXd energies_sol(L + 1);
  energies_sol << 6.17778, 2.92378, 2.62297, 2.55538, 2.52659, 2.50285;

  // Compute energies using user's function
  auto result = GuyerKrumhansl::track_GKHeatEnergy(T, L, M, mu0, zeta0, rho,
                                                   sigma, mu, kappa);

  // Check if the output of the user corresponds to the correct output
  EXPECT_EQ(energies_sol.size(), result.size());
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_LT(std::abs(result[i] - energies_sol[i]), 1e-5);
  }
}
}  // namespace GuyerKrumhansl::test
