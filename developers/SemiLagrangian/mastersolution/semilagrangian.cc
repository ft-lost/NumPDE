/**
 * @file
 * @brief NPDE homework TEMPLATE MAIN FILE
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "semilagrangian.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#define CSV_FORMAT Eigen::IOFormat(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"))


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
#if SOLUTION
  int N = u.size();  // assume dofs on boundary already removed
  int root = std::round(std::sqrt(N));
  int M = root + 1;
  double h = 1. / M;

  // compute the location of the square containing x
  int i = std::floor(x(0) / h);
  int j = std::floor(x(1) / h);

  // Check, if x is in the unit square
  if (i < 0 || i > M - 1 || j < 0 || j > M - 1) {
    std::cerr << "i,j can only be in [0,M-1]" << std::endl;
  }

  // compute local coordinates:
  Eigen::Vector2d x_loc;
  x_loc(0) = (x(0) - i * h) / h;
  x_loc(1) = (x(1) - j * h) / h;

  // Vector of local coefficients:
  Eigen::Vector4d u_loc;

  // Check for boundary dofs and extract correct coefficients of u:
  // Recall clockwise ordering of local dofs, starting in bottom left corner.
  u_loc(0) = (i == 0 || j == 0) ? 0.0 : u((M - 1) * (j - 1) + (i - 1));
  u_loc(1) = (i == (M - 1) || j == 0) ? 0.0 : u((M - 1) * (j - 1) + i);
  u_loc(2) = (i == (M - 1) || j == (M - 1)) ? 0.0 : u((M - 1) * j + i);
  u_loc(3) = (i == 0 || j == (M - 1)) ? 0.0 : u((M - 1) * j + i - 1);

  // evaluate using reference shape functions:
  return u_loc(0) * (1. - x_loc(0)) * (1. - x_loc(1)) +
         u_loc(1) * (1. - x_loc(1)) * x_loc(0) +
         u_loc(2) * x_loc(0) * x_loc(1) + u_loc(3) * (1. - x_loc(0)) * x_loc(1);
#else
  //====================
  // Your code goes here
  //====================
  return 0.0;
#endif
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

void SemiLagrangeVis(int M , int K , double T){
  std::vector<Eigen::VectorXd> u_t; // u_t will store the solution at every timestep
  // Create a recorder that can be passed to semiLagrangePureTransport
  auto rec= [&u_t](const Eigen::VectorXd & u){u_t.push_back(u);};
  semiLagrangePureTransport(M , K , T , rec); // Solve the pure transport equation
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(K,0,T);
  Eigen::VectorXd lin = Eigen::VectorXd::LinSpaced(M , 0 , 1);
  // Create Csv file out of u_h
  std::ofstream solution_file;
  solution_file.open("solution.csv");
  // Write the solution into a csv file
  for (auto& solution : u_t) {
    solution_file << solution.transpose().format(CSV_FORMAT) << std::endl;
  }
  solution_file.close();

    std::ostringstream oss;
    oss << "python3 " CURRENT_SOURCE_DIR
           "/make_gif.py " CURRENT_BINARY_DIR
           "/solution.csv " CURRENT_BINARY_DIR "/ "
            << M << " " << K << " " << T;

    std::string ostring = oss.str();
    const char* arguments = ostring.c_str() ;
    // Generating gif
    std::cout << "Creating gif" << std::endl;
    std::system("mkdir " CURRENT_BINARY_DIR "/img"); // Creates the directory which will hold the images

    std::system(arguments); // Executes the pythong plotting
}

}  // namespace SemiLagrangian
