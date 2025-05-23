/**
 * @file burgersequation_main.cc
 * @brief NPDE homework BurgersEquation code
 * @author Oliver Rietmann
 * @date 15.04.2019
 * @copyright Developed at ETH Zurich
 */

#include <Eigen/Core>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "burgersequation.h"
#include "systemcall.h"

int main() {
  /* SAM_LISTING_BEGIN_1 */
  const unsigned int N = 100;
  Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N + 1, -1.0, 4.0);
  Eigen::VectorXd mu03 = BurgersEquation::solveBurgersGodunov(0.3, N);
  Eigen::VectorXd mu30 = BurgersEquation::solveBurgersGodunov(3.0, N);

  // Write the solutions to a file that can be used for plotting.
#if SOLUTION
  std::ofstream solution_file;
  solution_file.open("solution.csv");
  solution_file << x.transpose().format(BurgersEquation::CSVFormat)
                << std::endl;
  solution_file << mu03.transpose().format(BurgersEquation::CSVFormat)
                << std::endl;
  solution_file << mu30.transpose().format(BurgersEquation::CSVFormat)
                << std::endl;
  solution_file.close();
  std::cout << "Generated solution.csv" << std::endl;
  systemcall::execute(
      "python3 ms_scripts/plot_solution.py solution.csv "
      "solution.eps");
#else
  //====================
  // Your code goes here
  //====================
#endif
  /* SAM_LISTING_END_1 */

  /* SAM_LISTING_BEGIN_2 */
  Eigen::Matrix<double, 3, 4> result = BurgersEquation::numexpBurgersGodunov();

  // Write the result to a file that can be used for plotting.
#if SOLUTION
  std::ofstream error_file;
  error_file.open("error.csv");
  error_file << result.format(BurgersEquation::CSVFormat) << std::endl;
  error_file.close();
  std::cout << "Generated error.csv" << std::endl;
  systemcall::execute("python3 ms_scripts/plot_error.py error.csv error.eps");
#else
  //====================
  // Your code goes here
  //====================
#endif
  /* SAM_LISTING_END_2 */

  return 0;
}
