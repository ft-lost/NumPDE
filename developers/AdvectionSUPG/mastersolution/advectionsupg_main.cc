/**
 * @file
 * @brief NPDE homework Streamline Upwind Method for Pure Advection Problem
 * @author R. Hiptmair
 * @date January 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Core>
#include <iostream>

#include "advectionsupg.h"

int main(int /*argc*/, char** /*argv*/) {
  AdvectionSUPG::cvgL2SUPG();

  return 0;

}
