/**
 * @ file
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author
 * @ date
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <string>

#include "asymptoticcvgfem.h"

int main(int argc, char** argv) {
  std::cout << "Study of asymptotic convergence of Lagrangian FEM" << std::endl;
  // Read number of refinement levels 
  int reflev = 4;
  if (argc > 1) {
    reflev = std::atoi(argv[1]);
  }
  Eigen::MatrixXd errs  = asymptoticcvgfem::studyAsymptoticCvg(reflev);
  std::cout << "Errors \n " << errs << std::endl;
  return 0;
}
