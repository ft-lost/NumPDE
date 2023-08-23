/**
 * @ file sufem_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author R. Hiptmair
 * @ date July 2023
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "sufem.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "Problem: Streamline-Upwind Galerkin FEM" << std::endl;
  SUFEM::testSUFEMConvergence(4, "rotation");
  return 0;
}
