/**
 * @ file fvpsystem_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "fvpsystem.h"
#include "systemcall.h"

/* SAM_LISTING_BEGIN_1 */
int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem FVPsystem by Wouter Tonnon\n";

  // Parameters of the simulation
  double a = -10;
  double b = 10;
  double T = 2;
  unsigned int N = 1000;
  unsigned int M = 1000;

  // We store the solution in this matrix, columns are timesteps
  Eigen::MatrixXd u1, u2;

  /* Your code goes here */

  // Write the solution to a txt-file.
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file;
  file.open("solution_u1.csv");
  for (int j = 0; j < u1.rows(); ++j)
    file << u1.row(j).format(CSVFormat) << std::endl;
  file.close();
  file.open("solution_u2.csv");
  for (int j = 0; j < u2.rows(); ++j)
    file << u2.row(j).format(CSVFormat) << std::endl;
  file.close();

  // Plot the solution using a python script
  systemcall::execute(
      "python3 ms_scripts/vis_solution.py solution_u1.csv solution_u1.eps u_1");

  systemcall::execute(
      "python3 ms_scripts/vis_solution.py solution_u2.csv solution_u2.eps u_2");

  return 0;
}
/* SAM_LISTING_END_1 */
