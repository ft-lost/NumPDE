/**
 * @ file guyerKrumhansl_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Ralf Hiptmair
 * @ date July 2023
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <cmath>
#include <iostream>

#include "guyerkrumhansl.h"

int main(int /*argc*/, char** /*argv*/) {
  GuyerKrumhansl::demo_EigenTripletInit(20);

  // Computation of energies
  /* SAM_LISTING_BEGIN_1 */
  const double T = 1.0;
  const double kappa = 1.0;
  const double mu = 0.1;
  const double rho = 1.0;
  const double sigma = 0.2;
  // Initial vectors
  const unsigned int M = 100;
  const unsigned int L = 100;
  Eigen::VectorXd mu0 =
      Eigen::sin(Eigen::ArrayXd::LinSpaced(M, 0.0, M_PI)).matrix();
  Eigen::VectorXd zeta0 = Eigen::VectorXd::Zero(M - 1);
  auto energies = GuyerKrumhansl::track_GKHeatEnergy(T, L, M, mu0, zeta0, rho,
                                                     sigma, mu, kappa);
  /* SAM_LISTING_END_1 */
  for (double Ek : energies) {
    std::cout << Ek << ", ";
  }
  std::cout << std::endl;
  return 0;
}
