/**
 * @file 1dwaveabsorbingbc_main.cc
 * @brief NPDE homework "1DWaveAbsorbingBC" code
 * @author Oliver Rietmann
 * @date 08.04.2019
 * @copyright Developed at ETH Zurich
 */

#include <Eigen/Core>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "1dwaveabsorbingbc.h"
#include "systemcall.h"

using namespace WaveAbsorbingBC1D;

const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                       Eigen::DontAlignCols, ", ", "\n");

int main() {
  /* SAM_LISTING_BEGIN_1 */
  double c = 1.0;
  double T = 7.0;
  unsigned int N = 100;
  unsigned int m = 2000;
  Eigen::MatrixXd R = waveLeapfrogABC(c, T, N, m);
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(m + 1, 0.0, T);

  // print the data, e.g. to a .csv file, in a suitable way
  std::ofstream solution_file;

  solution_file.open("solution.csv");
  Eigen::MatrixXd tR(m + 1, N + 2);
  tR.col(0) = t;
  tR.block(0, 1, m + 1, N + 1) = R;
  solution_file << tR.format(CSVFormat) << std::endl;
  solution_file.close();
  std::cout << "Generated solution.csv" << std::endl;
  systemcall::execute(
      "python3 ms_scripts/viswave.py solution.csv solution.eps");
  /* SAM_LISTING_END_1 */

  std::pair<Eigen::VectorXd, Eigen::VectorXd> energies =
      computeEnergies(R, c, T / m);
  Eigen::VectorXd E_pot = energies.first;
  Eigen::VectorXd E_kin = energies.second;

  /* SAM_LISTING_BEGIN_2 */
  std::ofstream energies_file;
  energies_file.open("energies.csv");
  energies_file << t.transpose().format(CSVFormat) << std::endl
                << E_pot.transpose().format(CSVFormat) << std::endl
                << E_kin.transpose().format(CSVFormat) << std::endl;
  energies_file.close();
  std::cout << "Generated energies.csv" << std::endl;
  systemcall::execute(
      "python3 ms_scripts/visenergies.py energies.csv "
      "energies.eps");
/* SAM_LISTING_END_2 */

  return 0;
}
