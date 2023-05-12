/**
 * @file
 * @brief NPDE homework TEMPLATE MAIN FILE
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "semilagrangian.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CSV_FORMAT \
  Eigen::IOFormat( \
      Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"))

namespace SemiLagrangian {

Eigen::MatrixXd findGrid(int M) {
  Eigen::MatrixXd grid(2, (M - 1) * (M - 1));

  double h = 1. / M;
  double x1 = h;

  for (int i = 0; i < M - 1; ++i) {
    double x0 = h;
    for (int j = 0; j < M - 1; ++j) {
      Eigen::Vector2d x;
      x << x0, x1;
      grid.col(i * (M - 1) + j) = x;
      x0 += h;
    }
    x1 += h;
  }
  return grid;
}

double evalFEfunction(const Eigen::Vector2d& x, const Eigen::VectorXd& u) {
  //====================
  // Your code goes here
  //====================
  return 0.0;
}

void testFloorAndDivision() {
  int M = 80;
  double h = 1.0 / 80;
  Eigen::Vector2d x(0.504, 0.1625);
  std::cout << "j: " << std::floor(x(1) / h) << "(exact: " << x(1) / h << ")"
            << std::endl;
  std::cout << "j*h: " << std::floor(x(1) / h) * h << std::endl;
  std::cout << "x_loc formula from the exercise (direct computation): "
            << (x(1) - std::floor(x(1) / h) * h) / h << std::endl;
  std::cout << "x_loc fmod: " << std::fmod(x(1), h) / h << std::endl;
  std::cout << "Backward transformation (exercise): "
            << std::floor(x(1) / h) * h +
                   ((x(1) - std::floor(x(1) / h) * h) / h) * h
            << std::endl;
  std::cout << "Backward transformation (fmod): "
            << std::floor(x(1) / h) * h + (std::fmod(x(1), h) / h) * h
            << std::endl;
}

void SemiLagrangeVis(int M, int K, double T) {
  //====================
  // Your code goes here
  //====================
}

}  // namespace SemiLagrangian
