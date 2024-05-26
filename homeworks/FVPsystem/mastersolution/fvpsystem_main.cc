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

  // Define the initial condition
  auto u0 = [](double x)->Eigen::Vector2d{
    Eigen::Vector2d out(2);
    if(x<=0) {
      out(0) = 1;
      out(1) = 1;
    }
    else {
      out(0) = 3;
      out(1) = 4;
    }
    return out;
  };

  // Recorder stores both elements of the vector solution in different matrices
  auto record = [&u1, &u2](Eigen::MatrixXd mu) -> void {
    u1.conservativeResize(u1.rows() + 1, mu.cols());
    u2.conservativeResize(u2.rows() + 1, mu.cols());
    //std::cout<< mu.rows() << mu.cols() << u1.rows()<<u1.cols()<<std::endl;
    u1.row(u1.rows()-1) = mu.row(0);
    u2.row(u2.rows()-1) = mu.row(1);
    return;
  };

  // Solve the system
  auto mu  = FVPsystem::ev1ExpPSystem(a,b,T,N,M,u0,record);


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
