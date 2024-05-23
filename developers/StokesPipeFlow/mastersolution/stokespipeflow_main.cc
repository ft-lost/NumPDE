/**
 * @ file stokespipeflow_main.cc
 * @ brief NPDE homework StokesPipeFlow MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>

#include "stokespipeflow.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem StokesPipeFlow by Wouter Tonnon (R)\n";
  // Read *.msh_file
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/pipe.msh");
  // print all physical entities:
  std::cout << "Physical Entities in Gmsh File " << std::endl;
  std::cout
      << "---------------------------------------------------------------\n";
  for (lf::base::dim_t codim = 0; codim <= 2; ++codim) {
    for (auto& pair : reader.PhysicalEntities(codim)) {
      std::cout << "codim = " << static_cast<int>(codim) << ": " << pair.first
                << " <=> " << pair.second << std::endl;
    }
  }
  std::cout << std::endl << std::endl;

  // Get Physical Entity Number from it's name (as specified in Gmsh):
  auto air_nr = reader.PhysicalEntityName2Nr("air");

  // Get all codim=0 entities that belong to the "air" physical entity:
  for (auto& e : reader.mesh()->Entities(0)) {
    if (reader.IsPhysicalEntity(*e, air_nr)) {
      // This entity belongs to the "air" physical entity.
    }
  }

  return 0;
}
