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
  /* Your code goes here */

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

  /* Your code goes here */
  energies = Eigen::MatrixXd(5, 2);

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
