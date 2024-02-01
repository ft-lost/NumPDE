/**
 * @file asymptoticcvgfem.h
 * @brief NPDE homework AsymptoticCvgFEM code
 * @author R. Hiptmair
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "semilinearellipticbvp.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE HW problem on a semin-linear elliptic BVP" << std::endl;

  semilinearellipticbvp::testSolverSemilinearBVP();

  return 0;
}
