/**
 * @file sdirk_main.cc
 * @brief NPDE homework SDIRK code
 * @author Unknown, Oliver Rietmann
 * @date 31.03.2021
 * @copyright Developed at ETH Zurich
 */

#include <cstdlib>
#include <iostream>

#include "sdirk.h"
#include "systemcall.h"

int main() {
  // Compute convergence rates
  double rate = SDIRK::CvgSDIRK();
  std::cout << std::endl << "The rate is " << rate << std::endl;

  // Plot stability domain for gamma = 1.0
  systemcall::execute("python3 scripts/stabdomSDIRK.py stabdomSDIRK.eps");
  std::cout << "Generated stabdomSDIRK.eps" << std::endl;

  return 0;
}
