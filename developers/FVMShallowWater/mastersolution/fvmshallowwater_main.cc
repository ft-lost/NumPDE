/**
 * @ file fvmshallowwater_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Ralf Hiptmair
 * @ date June 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <cstddef>
#include <fstream>
#include <iostream>

#include "fvmshallowwater.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem FVMShallowWater\n";

  // Riemann problem for shallow water equations (dam break problem)
  auto u0 = [](double x) -> Eigen::Vector2d {
    if (x < 0.0) {
      return {1.0, 0.0};
    }
    return {3.0, 0.0};
  };
  double T = 0.5;
  size_t N = 200;
  {
    std::ofstream file("LFsol.m");
    (void)FVMShallowWater::solveSWE(T, N, &FVMShallowWater::numfluxLFSWE, u0,
                                    &file);
  }
  {
    std::ofstream file("HLLEsol.m");
    (void)FVMShallowWater::solveSWE(T, N, &FVMShallowWater::numfluxHLLESWE, u0,
                                    &file);
  }
  return 0;
}
