/**
 * @ file stokespipeflow_main.cc
 * @ brief NPDE homework StokesPipeFlow MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <lf/base/lehrfem_info.h>
#include <lf/base/ref_el.h>

#include <iostream>
#include <memory>

#include "stokespipeflow.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem StokesPipeFlow by Wouter Tonnon (R)\n";
  lf::base::LehrFemInfo::PrintInfo(std::cout);

  // Test of convergence on the unit square with manufactured solution
//  StokesPipeFlow::testCvgTaylorHood(4);
  // Compute on "realistic" geometry
//  std::cout << "Writing vtk-File" << std::endl;
  StokesPipeFlow::visualizeTHPipeFlow("meshes/pipe.msh", "out.vtk");
//  double p_diss_vol = StokesPipeFlow::computeDissipatedPower("meshes/pipe.msh");
//  double p_diss_bd =
//      StokesPipeFlow::computeDissipatedPowerBd("meshes/pipe.msh");
//  std::cout << "Dissipated power: volume formula = " << p_diss_vol
//            << ", boundary formula = " << p_diss_bd << std::endl;
  return 0;
}
