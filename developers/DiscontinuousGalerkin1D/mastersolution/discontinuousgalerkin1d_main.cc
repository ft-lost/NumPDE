/**
 * @file discontinuousgalerkin1d_main.cc
 * @brief NPDE homework "DiscontinuousGalerkin1D" code
 * @author Oliver Rietmann
 * @date 22.05.2019
 * @copyright Developed at ETH Zurich
 */

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "discontinuousgalerkin1d.h"
#include "systemcall.h"

int main() {
  DiscontinuousGalerkin1D::Solution solution =
      DiscontinuousGalerkin1D::solveTrafficFlow();

  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");

#if SOLUTION
  std::ofstream file;
  file.open("solution.csv");
  file << solution.x_.transpose().format(CSVFormat) << std::endl;
  file << solution.u_.transpose().format(CSVFormat) << std::endl;
  file.close();

  std::cout << "Generated solution.csv" << std::endl;
  systemcall::execute(
      "python3 scripts/plot_solution.py solution.csv solution.eps");
#else
  //====================
  // Your code goes here
  // Use std::ofstream to write the solution to
  // the file "solution.csv". To plot this file
  // you may uncomment the following line:
  // systemcall::execute("python3 scripts/plot_solution.py solution.csv
  // solution.eps");
  //====================
#endif

  return 0;
}
