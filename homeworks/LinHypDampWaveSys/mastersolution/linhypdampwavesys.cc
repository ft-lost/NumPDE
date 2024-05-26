/**
 * @file linhypdampwavesys.cc
 * @brief NPDE homework LinHypDampWaveSys code
 * @author Wouter Tonnon
 * @date May 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#include "linhypdampwavesys.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

namespace LinHypDampWaveSys {

/* SAM_LISTING_BEGIN_1 */
void visWavSol(double c, double r, double T, unsigned int N,
               Fluxtype nf_selector) {
  // Store the solution in this matrix, every column is a timestep
  Eigen::MatrixXd p;
  // Define the initial condition
  auto u0 = [](double x) -> Eigen::Vector2d {
    Eigen::Vector2d out;
    out(0) = 0.;
    if (0 <= x && x <= .5)
      out(1) = 1.;
    else if (0.5 <= x && x <= 1)
      out(1) = -1.;
    else
      out(1) = 0.;
    return out;
  };

  // Some parameters of the numerical method
  double maxspeed = c;
  double a = -T * maxspeed;
  double b = 1 + T * maxspeed;
  double h = (b - a) / N;

  // Recorder that automatically constructs p from the state variable for dp/dx
  auto record = [&p, h](Eigen::MatrixXd mu) -> void {
    p.conservativeResize(p.rows() + 1, mu.cols() + 1);
    Eigen::VectorXd dpdx = h * mu.row(1);
    p(p.rows() - 1, 0) = 0.;
    std::partial_sum(dpdx.begin(), dpdx.end(), p.row(p.rows() - 1).begin() + 1,
                     std::plus<double>());
    return;
  };

  // Execute the solver
  ev1DampWave(c, r, T, N, u0, nf_selector, record);


  // Write the solution to a txt-file.
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file;
  file.open("solution.csv");
  for (int j = 0; j < p.rows(); ++j)
    file << p.row(j).format(CSVFormat) << std::endl;
  file.close();

  // Return
  return;
}

/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
Eigen::Matrix<double, Eigen::Dynamic, 2> trackEnergysWavSol(
    double c, double r, double T, unsigned int N, Fluxtype nf_selector) {
  // Store energies in the following matrix
  Eigen::Matrix<double, Eigen::Dynamic, 2> energies;

  // Define the initial condition
  auto u0 = [](double x) -> Eigen::Vector2d {
    Eigen::Vector2d out;
    out(0) = 0.;
    if (0 <= x && x <= .5)
      out(1) = 1.;
    else if (0.5 <= x && x <= 1)
      out(1) = -1.;
    else
      out(1) = 0.;
    return out;
  };

  // Some parameters of the numerical method
  double maxspeed = c;
  double a = -T * maxspeed;
  double b = 1 + T * maxspeed;
  double h = (b - a) / N;
  double dt = h / maxspeed;  //*(b - a) / ((N * maxspeed));

  // Recorder that computes energy
  auto record = [&energies, c, dt](Eigen::MatrixXd mu) -> void {
    unsigned int i = energies.rows();
    energies.conservativeResize(energies.rows() + 1, 2);
    energies(i, 0) = dt * i;
    energies(i, 1) =
        mu.row(0).dot(mu.row(0)) + c * c * mu.row(1).dot(mu.row(1));
    return;
  };

  // Execute the solver
  ev1DampWave(c, r, T, N, u0, nf_selector, record);


  // Write to txt-file
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file;
  file.open("energies.csv");
  file << energies.col(0).transpose().format(CSVFormat) << std::endl;
  file << energies.col(1).transpose().format(CSVFormat) << std::endl;
  file.close();

  // Return
  return energies;
}
/* SAM_LISTING_END_2 */

}  // namespace LinHypDampWaveSys
