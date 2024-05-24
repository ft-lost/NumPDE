/**
 * @file StokesPipeFlow_test.cc
 * @brief NPDE homework StokesPipeFlow code
 * @author R. Hiptmair
 * @date May 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../stokespipeflow.h"

#include <gtest/gtest.h>
#include <lf/base/ref_el.h>
#include <lf/fe/loc_comp_ellbvp.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/test_utils/test_meshes.h>

#include <Eigen/Core>
#include <cstddef>
#include <memory>

namespace StokesPipeFlow::test {

// Copied from lecturedemodof.cc
void printDofInfo(const lf::assemble::DofHandler& dofh) {
  // Obtain pointer to the underlying mesh
  auto mesh = dofh.Mesh();
  // Number of degrees of freedom managed by the DofHandler object
  const lf::assemble::size_type N_dofs(dofh.NumDofs());
  std::cout << "DofHandler(" << dofh.NumDofs() << " dofs):" << '\n';
  // Output information about dofs for entities of all co-dimensions
  for (lf::base::dim_t codim = 0; codim <= mesh->DimMesh(); codim++) {
    // Visit all entities of a codimension codim
    for (const lf::mesh::Entity* e : mesh->Entities(codim)) {
      // Fetch unique index of current entity supplied by mesh object
      const lf::base::glb_idx_t e_idx = mesh->Index(*e);
      // Number of shape functions covering current entity
      const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*e));
      // Obtain global indices of those shape functions ...
      const std::span<const lf::assemble::gdof_idx_t> dofarray{
          dofh.GlobalDofIndices(*e)};
      // and print them
      std::cout << *e << ' ' << e_idx << ": " << no_dofs << " dofs = [";
      for (int loc_dof_idx = 0; loc_dof_idx < no_dofs; ++loc_dof_idx) {
        std::cout << dofarray[loc_dof_idx] << ' ';
      }
      std::cout << ']';
      // Also output indices of interior shape functions
      const std::span<const lf::assemble::gdof_idx_t> intdofarray{
          dofh.InteriorGlobalDofIndices(*e)};
      std::cout << " int = [";
      for (const lf::assemble::gdof_idx_t int_dof : intdofarray) {
        std::cout << int_dof << ' ';
      }
      std::cout << ']' << '\n';
    }
  }
  // List entities associated with the dofs managed by the current
  // DofHandler object
  for (lf::assemble::gdof_idx_t dof_idx = 0; dof_idx < N_dofs; dof_idx++) {
    const lf::mesh::Entity& e(dofh.Entity(dof_idx));
    std::cout << "dof " << dof_idx << " -> " << e << " " << mesh->Index(e)
              << '\n';
  }
}  // end function printDofInfo

// Generate "mesh" consisting of three triangles
// Partly copied from lecturedemodoc.cc
std::shared_ptr<lf::mesh::Mesh> getThreeTriagMesh() {
  // Short name for 2d coordinate vectors
  using coord_t = Eigen::Vector2d;
  // Corner coordinates for a triangle
  using tria_coord_t = Eigen::Matrix<double, 2, 3>;
  // Ftehc a mesh facvtory
  const std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  // Add points
  mesh_factory_ptr->AddPoint(coord_t({0, 0}));    // point 0
  mesh_factory_ptr->AddPoint(coord_t({1, 0}));    // point 1
  mesh_factory_ptr->AddPoint(coord_t({0, 1}));    // point 2
  mesh_factory_ptr->AddPoint(coord_t({1, 1}));  // point 3
  mesh_factory_ptr->AddPoint(coord_t({1, 2}));  // point 4
  
  // Define triangular cells
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({0, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({3, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({2, 3, 4}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  // Ready to build the mesh data structure
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();
  return mesh_p;
}

// Generate "mesh" consisting of four triangles with an interior vertex
// Partly copied from lecturedemodoc.cc
std::shared_ptr<lf::mesh::Mesh> getFourTriagMesh() {
  // Short name for 2d coordinate vectors
  using coord_t = Eigen::Vector2d;
  // Corner coordinates for a triangle
  using tria_coord_t = Eigen::Matrix<double, 2, 3>;
  // Ftehc a mesh facvtory
  const std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  // Add points
  mesh_factory_ptr->AddPoint(coord_t({0, 0}));    // point 0
  mesh_factory_ptr->AddPoint(coord_t({1, 0}));    // point 1
  mesh_factory_ptr->AddPoint(coord_t({1, 1}));    // point 2
  mesh_factory_ptr->AddPoint(coord_t({0, 1}));    // point 3
  mesh_factory_ptr->AddPoint(coord_t({0.3, 0.7}));  // point 4
  
  // Define triangular cells
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({0, 1, 4}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({1, 2, 4}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({2, 3, 4}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({3, 0, 4}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  // Ready to build the mesh data structure
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();
  return mesh_p;
}
  
  
// Generate "mesh" consisting of two triangles
// Partly copied from lecturedemodoc.cc
std::shared_ptr<lf::mesh::Mesh> getTwoTriagMesh() {
  // Short name for 2d coordinate vectors
  using coord_t = Eigen::Vector2d;
  // Corner coordinates for a triangle
  using tria_coord_t = Eigen::Matrix<double, 2, 3>;
  // Ftehc a mesh facvtory
  const std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  // Add points
  mesh_factory_ptr->AddPoint(coord_t({0, 0}));      // point 0
  mesh_factory_ptr->AddPoint(coord_t({1, 0}));      // point 1
  mesh_factory_ptr->AddPoint(coord_t({0, 1}));      // point 2
  mesh_factory_ptr->AddPoint(coord_t({0.9, 0.7}));  // point 3
  // Define triangular cells
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({0, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({3, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  // Ready to build the mesh data structure
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();
  return mesh_p;
}

// Test whether locally constant velocity fields/pressures are in the
// kernel of the element matrices
bool testKernelTHElMat(const lf::mesh::Mesh& mesh, bool print = false) {
  // Set up ENTITY MATRIX PROVIDER
  StokesPipeFlow::TaylorHoodElementMatrixProvider themp{};
  // Test vectors
  Eigen::VectorXd ldofs(15);
  Eigen::VectorXd vdofs{Eigen::VectorXd::Zero(15)};
  Eigen::VectorXd pdofs{Eigen::VectorXd::Zero(15)};
  // x-component of the velocity
  ldofs[0] = ldofs[3] = ldofs[6] = ldofs[9] = ldofs[11] = ldofs[13] = 1.23;
  // y-component of the velocity
  ldofs[1] = ldofs[4] = ldofs[7] = ldofs[10] = ldofs[12] = ldofs[14] = 4.56;
  // pressure dofs (constant pressure not in the kernel)
  ldofs[2] = ldofs[5] = ldofs[8] = 0.0;
  // For second test
  vdofs = ldofs;
  vdofs[9] = 0.5 * (vdofs[0] + vdofs[3]);
  vdofs[11] = 0.5 * (vdofs[3] + vdofs[6]);
  vdofs[13] = 0.5 * (vdofs[0] + vdofs[6]);
  vdofs[10] = 0.5 * (vdofs[1] + vdofs[4]);
  vdofs[12] = 0.5 * (vdofs[4] + vdofs[7]);
  vdofs[14] = 0.5 * (vdofs[1] + vdofs[7]);
  pdofs[2] = 1.0;
  pdofs[5] = -1.0 / 3.0;
  pdofs[8] = -2.0 / 3.0;
  // Loop over cells and fetch element matrices
  bool res = false;
  for (const lf::mesh::Entity* cell : mesh.Entities(0)) {
    EXPECT_EQ(cell->RefEl(), lf::base::RefEl::kTria());
    // Compute element matrix
    const StokesPipeFlow::TaylorHoodElementMatrixProvider::ElemMat AK(
        themp.Eval(*cell));
    // Print on demand
    if (print) {
      std::cout << "Triangular cell:\n"
                << lf::geometry::Corners(*(cell->Geometry())) << "\n";
      std::cout << "Element matrix\n" << AK << std::endl;
    }
    // First test: constants in the kernel
    res = ((AK * ldofs).norm() < 1E-10);
    EXPECT_TRUE(res);
    // Second test: Linear velocity, zero average pressure
    EXPECT_NEAR(pdofs.transpose() * AK * vdofs, 0.0, 1E-10);
  }
  return res;
}

void compareVelocityElMats(const std::shared_ptr<const lf::mesh::Mesh> mesh_p,
                           bool print = false) {
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto one = [](const Eigen::Vector2d& /*x*/) -> double { return 1.0; };
  const lf::mesh::utils::MeshFunctionGlobal mf_one{one};
  // Set up ENTITY MATRIX PROVIDERs
  StokesPipeFlow::TaylorHoodElementMatrixProvider themp{};
  lf::fe::DiffusionElementMatrixProvider Laplace_emp(fe_space, mf_one);
  // Local indices for dofs
  const std::array<Eigen::Index, 6> vx_idx{0, 3, 6, 9, 11, 13};
  const std::array<Eigen::Index, 6> vy_idx{1, 4, 7, 10, 12, 14};
  const std::array<Eigen::Index, 3> p_idx{2, 5, 8};
  // Loop over cells
  for (const lf::mesh::Entity* cell : mesh_p->Entities(0)) {
    EXPECT_EQ(cell->RefEl(), lf::base::RefEl::kTria());
    // Compute element matrix
    const StokesPipeFlow::TaylorHoodElementMatrixProvider::ElemMat AK(
        themp.Eval(*cell));
    // Rearrange element matrix
    StokesPipeFlow::TaylorHoodElementMatrixProvider::ElemMat As;
    As.setZero();
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        As(i, j) = As(j, i) = AK(vx_idx[i], vx_idx[j]);
        As(i + 6, j + 6) = As(j + 6, i + 6) = AK(vy_idx[i], vy_idx[j]);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        As(12 + i, j) = As(j, i + 12) = AK(p_idx[i], vx_idx[j]);
        As(12 + i, j + 6) = As(j + 6, i + 12) = AK(p_idx[i], vy_idx[j]);
      }
    }
    // Element matrix for degree 2 Lagrangian FEM
    auto L2A = Laplace_emp.Eval(*cell);
    // Print on demand
    if (print) {
      std::cout << "Triangular cell:\n"
                << lf::geometry::Corners(*(cell->Geometry())) << "\n";
      std::cout << "Reordered element matrix\n" << As << std::endl;
      std::cout << "Element matrix for quadrative Lagrangian FEM\n"
                << L2A << std::endl;
    }
    EXPECT_NEAR((As.block(0, 0, 6, 6) - L2A).norm(), 0.0, 1E-10);
    EXPECT_NEAR((As.block(6, 6, 6, 6) - L2A).norm(), 0.0, 1E-10);
  }
}

TEST(StokesPipeFlow, TaylorHoodElementMatrixProvider) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = getTwoTriagMesh();
  EXPECT_TRUE(testKernelTHElMat(*mesh_p, false));
  compareVelocityElMats(mesh_p, true);
}

TEST(StokesPipeFlow, THEMP_complex) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  EXPECT_TRUE(testKernelTHElMat(*mesh_p, false));
  compareVelocityElMats(mesh_p, false);
}

TEST(StokesPipeFlow, DISABLED_PrintDof) {
std::shared_ptr<const lf::mesh::Mesh> mesh_p = getTwoTriagMesh();
// Set up DofHandler
lf::assemble::UniformFEDofHandler dofh(mesh_p,
                                       {{lf::base::RefEl::kPoint(), 3},
                                        {lf::base::RefEl::kSegment(), 2},
                                        {lf::base::RefEl::kTria(), 0},
                                        {lf::base::RefEl::kQuad(), 0}});
printDofInfo(dofh);
}


TEST(StokesPipeFlow, GalerkinMatrix) {
  // Obtain a purely triangular mesh from the collection of LehrFEM++'s
  // built-in meshes

  /*
  const std::shared_ptr<lf::mesh::Mesh> mesh_p{
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0)};
  */
  /* std::shared_ptr<const lf::mesh::Mesh> mesh_p = getTwoTriagMesh(); */
  // std::shared_ptr<const lf::mesh::Mesh> mesh_p = getThreeTriagMesh();
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = getFourTriagMesh();
  // Taylor Hood FEM
  // Set up DofHandler
  lf::assemble::UniformFEDofHandler dofh(mesh_p,
                                         {{lf::base::RefEl::kPoint(), 3},
                                          {lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});
  // Total number of FE d.o.f.s without Lagrangian multiplier
  lf::assemble::size_type n = dofh.NumDofs();
  // Full Galerkin matrix in triplet format taking into account the zero mean
  // constraint on the pressure.
  lf::assemble::COOMatrix<double> A(n, n);
  // Set up computation of element matrix
  TaylorHoodElementMatrixProvider themp{};
  // Assemble \cor{full} Galerkin matrix for Taylor-Hood FEM
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, themp, A);
  // Convert into a dense matrix
  Eigen::MatrixXd A_dense = A.makeDense();
  LF_ASSERT_MSG((A_dense.cols() == n) && (A_dense.rows() == n),
                "Wrong size of Galerkin matrix");

  // Quadratic Lagrange FEM
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto one = [](const Eigen::Vector2d& /*x*/) -> double { return 1.0; };
  const lf::mesh::utils::MeshFunctionGlobal mf_one{one};
  lf::fe::DiffusionElementMatrixProvider Laplace_emp(fe_space, mf_one);
  const lf::assemble::DofHandler& dofh_LO2{fe_space->LocGlobMap()};
  lf::assemble::size_type n_LO2 = dofh_LO2.NumDofs();
  lf::assemble::COOMatrix<double> A_LO2(n_LO2, n_LO2);
  lf::assemble::AssembleMatrixLocally(0, dofh_LO2, dofh_LO2, Laplace_emp,
                                      A_LO2);
  Eigen::MatrixXd Ad_LO2 = A_LO2.makeDense();

  // Indices for velocity and pressure dofs
  std::vector<Eigen::Index> vx_idx;
  std::vector<Eigen::Index> vy_idx;
  std::vector<Eigen::Index> LO2_idx;
  std::vector<Eigen::Index> p_idx;
  std::vector<Eigen::Index> bdv_idx;
  // Set rows/cols corresponding to dofs on the boundary to zero
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p)};
  for (const lf::mesh::Entity* node : mesh_p->Entities(2)) {
    std::span<const lf::assemble::gdof_idx_t> dof_idx{
        dofh.InteriorGlobalDofIndices(*node)};
    std::span<const lf::assemble::gdof_idx_t> LO2_ldx{
        dofh_LO2.InteriorGlobalDofIndices(*node)};
    LF_ASSERT_MSG(dof_idx.size() == 3, "Node must carry 3 dofs!");
    LF_ASSERT_MSG(LO2_ldx.size() == 1, "LO2: Node carries only a single dof!");
    vx_idx.push_back(dof_idx[0]);
    vy_idx.push_back(dof_idx[1]);
    p_idx.push_back(dof_idx[2]);
    LO2_idx.push_back(LO2_ldx[0]);
    if (bd_flags(*node)) {
      bdv_idx.push_back(dof_idx[0]);
      bdv_idx.push_back(dof_idx[1]);
    }
  }
  for (const lf::mesh::Entity* edge : mesh_p->Entities(1)) {
    std::span<const lf::assemble::gdof_idx_t> dof_idx{
        dofh.InteriorGlobalDofIndices(*edge)};
    std::span<const lf::assemble::gdof_idx_t> LO2_ldx{
        dofh_LO2.InteriorGlobalDofIndices(*edge)};
    LF_ASSERT_MSG(dof_idx.size() == 2, "Edge must carry 2 dofs!");
    LF_ASSERT_MSG(LO2_ldx.size() == 1, "LO2: Edge carries a single dof");
    vx_idx.push_back(dof_idx[0]);
    vy_idx.push_back(dof_idx[1]);
    LO2_idx.push_back(LO2_ldx[0]);
    if (bd_flags(*edge)) {
      bdv_idx.push_back(dof_idx[0]);
      bdv_idx.push_back(dof_idx[1]);
    }
  }
  // Test dimension count
  EXPECT_EQ(vx_idx.size() + vy_idx.size() + p_idx.size(), n);
  EXPECT_EQ(vx_idx.size(), LO2_idx.size());
  EXPECT_EQ(vy_idx.size(), LO2_idx.size());

  // Test Laplacian matrices
  for (int i = 0; i < LO2_idx.size(); ++i) {
    for (int j = 0; j < LO2_idx.size(); ++j) {
      EXPECT_NEAR(Ad_LO2(LO2_idx[i], LO2_idx[j]), A_dense(vx_idx[i], vx_idx[j]),
                  1E-10);
      EXPECT_NEAR(Ad_LO2(LO2_idx[i], LO2_idx[j]), A_dense(vy_idx[i], vy_idx[j]),
                  1E-10);
    }
  }

  // Print reordered matrices
  size_t n_vx = vx_idx.size();
  size_t n_p = p_idx.size();
  Eigen::MatrixXd Ar(A.rows(),A.cols());
  Ar.setZero();
  for (int i = 0; i < n_vx; ++i) {
    for (int j = 0; j < n_vx; ++j) {
      Ar(i,j) = A_dense(vx_idx[i],vx_idx[j]);
      Ar(i+n_vx,j+n_vx) = A_dense(vy_idx[i],vy_idx[j]);
    }
  }
   for (int i = 0; i < n_p; ++i) {
     for (int j = 0; j < n_vx; ++j) {
       Ar(2*n_vx+i,j) =  Ar(j, 2*n_vx+i) = A_dense(p_idx[i],vx_idx[j]);
       Ar(2*n_vx+i,j+n_vx) =  Ar(j+n_vx, 2*n_vx+i) = A_dense(p_idx[i],vy_idx[j]);
     }
   }
   std::cout << "Full Galerkin matrix: REORDERED\n " << Ar << std::endl;
   std::cout << "Full LO2 Galerkin matrix\n" << Ad_LO2 << std::endl;
  
  // Test whether contant velocities are in the kernel of the matrix
  Eigen::VectorXd dofs(n);
  for (int k = 0; k < vx_idx.size(); ++k) {
    dofs[vx_idx[k]] = 1.23;
  }
  for (int k = 0; k < vy_idx.size(); ++k) {
    dofs[vy_idx[k]] = 4.56;
  }
  for (int k = 0; k < p_idx.size(); ++k) {
    dofs[p_idx[k]] = 0.0;
  }
  EXPECT_NEAR((A_dense * dofs).norm(), 0.0, 1E-10);
}

TEST(StokesPipeFlow, solvePipeFlow) {
  // Populate Galerkin Matrix
}

}  // namespace StokesPipeFlow::test
