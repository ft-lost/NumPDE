/**
 * @ file stokespipeflow_main.cc
 * @ brief NPDE homework StokesPipeFlow MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>

#include "stokespipeflow.h"

int main(int /*argc*/, char ** /*argv*/) {
  std::cout << "NumPDE homework problem StokesPipeFlow by Wouter Tonnon (R)\n";
  // Read *.msh_file
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/pipe.msh");

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh &mesh{*mesh_ptr};

  auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);

  // Initialize dof handler
  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};

  const lf::base::size_type N_dofs(dofh.NumDofs());

  lf::io::VtkWriter vtk_writer(mesh_ptr, "test.vtk");
  auto u1 = lf::mesh::utils::MeshFunctionGlobal([](const Eigen::Vector2d& x) {
    return -x[1];
  });  
  auto u2 = lf::mesh::utils::MeshFunctionGlobal([](const Eigen::Vector2d& x) {
    return x[0];
  });

  vtk_writer.WritePointData("u1", u1);
  vtk_writer.WritePointData("u2", u2);
  vtk_writer.WritePointData("p", u1);
 

/*
  auto cell_data_ref =
      lf::mesh::utils::make_CodimMeshDataSet<double>(mesh_p, 0);
  for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
    int row = dofh.GlobalDofIndices(*cell)[0];
    cell_data_ref->operator()(*cell) = solution[row];
  }
  vtk_writer.WriteCellData(name, *cell_data_ref);
*/
  return 0;
}
