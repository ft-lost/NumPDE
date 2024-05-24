/**
 * @ file stokespipeflow_main.cc
 * @ brief NPDE homework StokesPipeFlow MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>
#include <memory>

#include "stokespipeflow.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem StokesPipeFlow by Wouter Tonnon (R)\n";
  // Read *.msh_file
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/simple.msh");


  StokesPipeFlow::allPipeFlow(StokesPipeFlow::VOLUME, true, "meshes/simple.msh", "out.vtk");

  return 0;
}
