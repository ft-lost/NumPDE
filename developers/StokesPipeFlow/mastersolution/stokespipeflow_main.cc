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

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh& mesh{*mesh_ptr};

  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);

  // Initialize dof handler
  lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                         {{lf::base::RefEl::kPoint(), 3},
                                          {lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});

  // Initialize dof handler
  lf::assemble::UniformFEDofHandler dofh_u(mesh_ptr,
                                           {{lf::base::RefEl::kPoint(), 1},
                                            {lf::base::RefEl::kSegment(), 1},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});

  // Initialize dof handler
  lf::assemble::UniformFEDofHandler dofh_p(mesh_ptr,
                                           {{lf::base::RefEl::kPoint(), 1},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});

  auto g = [](Eigen::VectorXd x) -> Eigen::VectorXd {
    Eigen::VectorXd out(2);
    out(0) = 1.;
    out(1) = 0.;
    return out;
  };

  auto res = StokesPipeFlow::solvePipeFlow(dofh, g);
  std::cout << res << std::endl << std::endl;

  int numVertices = mesh.NumEntities(lf::base::RefEl::kPoint());
  int numEdges = mesh.NumEntities(lf::base::RefEl::kPoint());

  Eigen::VectorXd coeff_vec_u1(dofh_u.NumDofs());
  Eigen::VectorXd coeff_vec_u2(dofh_u.NumDofs());
  for (auto e : mesh.Entities(0)) {
    auto glob_idxs = dofh.GlobalDofIndices(*e);

    auto glob_idx_o2 = dofh.GlobalDofIndices(*e)[0];

    coeff_vec_u1(glob_idx_o2) = res(glob_idxs[0]);
    coeff_vec_u2(glob_idx_o2) = res(glob_idxs[1]);

  }
  for (int i = 0; i < numEdges; ++i) {
    coeff_vec_u1(numVertices + i) = res(3 * numVertices + i * 2);
    coeff_vec_u1(numVertices + i) = res(3 * numVertices + i * 2 + 1);
  }

  auto fes_o2_p =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);

  lf::fe::MeshFunctionFE<double, double> mf_o2_u1(fes_o2_p, coeff_vec_u1);
  lf::fe::MeshFunctionFE<double, double> mf_o2_u2(fes_o2_p, coeff_vec_u2);

  // lf::fe::fe::MeshFUnctionFE<double,double> mf_o2()

  lf::io::VtkWriter vtk_writer(mesh_ptr, "test.vtk");

  vtk_writer.WritePointData("u1", mf_o2_u1);
  vtk_writer.WritePointData("u2", mf_o2_u2);
  // vtk_writer.WritePointData("p", u1);

  /*
    auto u1 = lf::mesh::utils::MeshFunctionGlobal(
        [](const Eigen::Vector2d& x) { return -x[1]; });
    auto u2 = lf::mesh::utils::MeshFunctionGlobal(
        [](const Eigen::Vector2d& x) { return x[0]; });
    const lf::base::size_type N_dofs(dofh.NumDofs());

    lf::io::VtkWriter vtk_writer(mesh_ptr, "test.vtk");
    auto u1 = lf::mesh::utils::MeshFunctionGlobal(
        [](const Eigen::Vector2d& x) { return -x[1]; });
    auto u2 = lf::mesh::utils::MeshFunctionGlobal(
        [](const Eigen::Vector2d& x) { return x[0]; });

    vtk_writer.WritePointData("u1", u1);
    vtk_writer.WritePointData("u2", u2);
    vtk_writer.WritePointData("p", u1);
    */

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
