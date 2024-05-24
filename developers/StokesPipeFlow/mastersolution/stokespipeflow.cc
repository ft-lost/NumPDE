/**
 * @file stokespipeflow.cc
 * @brief NPDE homework StokesPipeFlow code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "stokespipeflow.h"

#include <Eigen/src/Core/util/Meta.h>
#include <lf/assemble/assembler.h>
#include <lf/assemble/assembly_types.h>
#include <lf/assemble/coomatrix.h>
#include <lf/base/lf_assert.h>
#include <lf/base/ref_el.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/entity.h>

#include <cstddef>

namespace StokesPipeFlow {
/* SAM_LISTING_BEGIN_1 */
TaylorHoodElementMatrixProvider::ElemMat TaylorHoodElementMatrixProvider::Eval(
    const lf::mesh::Entity& cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions, see
  // \lref{cpp:gradbarycoords} Get vertices of the triangle
  auto endpoints = lf::geometry::Corners(*(cell.Geometry()));
  Eigen::Matrix<double, 3, 3> X;  // temporary matrix
  X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
  X.block<3, 2>(0, 1) = endpoints.transpose();
  // Thios matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
  // Dummy lambda functions for barycentric coordinates
  std::array<std::function<double(Eigen::Vector3d)>, 3> lambda{
      [](Eigen::Vector3d c) -> double { return c[0]; },
      [](Eigen::Vector3d c) -> double { return c[1]; },
      [](Eigen::Vector3d c) -> double { return c[2]; }};
  // Gradients of local shape functions of quadratic Lagrangian finite element
  // space as lambda functions, see \prbeqref{eq:quadlsf}
  std::array<std::function<Eigen::Vector2d(Eigen::Vector3d)>, 6> gradq{
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return (4 * c[0] - 1) * G.col(0);
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return (4 * c[1] - 1) * G.col(1);
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return (4 * c[2] - 1) * G.col(2);
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return 4 * (c[0] * G.col(1) + c[1] * G.col(0));
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return 4 * (c[1] * G.col(2) + c[2] * G.col(1));
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return 4 * (c[2] * G.col(0) + c[0] * G.col(2));
      }};
  // Barycentric coordinates of the midpoints of the edges for
  // use with the 3-point edge midpoint quadrature rule \prbeqref{eq:MPR}
  const std::array<Eigen::Vector3d, 3> mp = {Eigen::Vector3d({0.5, 0.5, 0}),
                                             Eigen::Vector3d({0, 0.5, 0.5}),
                                             Eigen::Vector3d({0.5, 0, 0.5})};
  // Compute the (scaled) element matrix  for $-\Delta$ and $\cob{\Cs^0_2}$.
  Eigen::Matrix<double, 6, 6> L;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      // 3-point edge midpoint quadrature rule!
      L(i, j) = L(j, i) = gradq[i](mp[0]).dot(gradq[j](mp[0])) +
                          gradq[i](mp[1]).dot(gradq[j](mp[1])) +
                          gradq[i](mp[2]).dot(gradq[j](mp[2]));
    }
  }
  // Do not forget to set all non-initialized entries to zero
  AK_.setZero();
  // Distribute the entries of L to the final element matrix
  const std::array<Eigen::Index, 6> vx_idx{0, 3, 6, 9, 11, 13};
  const std::array<Eigen::Index, 6> vy_idx{1, 4, 7, 10, 12, 14};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      AK_(vx_idx[i], vx_idx[j]) = AK_(vy_idx[i], vy_idx[j]) = L(i, j);
    }
  }
  // Fill entries related to bilinear form b(.,.): \prbeqref{eq:BKent}
  const std::array<Eigen::Index, 3> p_idx{2, 5, 8};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 6; ++j) {
      // \prbeqref{eq:BKvec} with 3-point edge midpoint quadrature rule!
      const Eigen::Vector2d gql_ij{gradq[j](mp[0]) * lambda[i](mp[0]) +
                                   gradq[j](mp[1]) * lambda[i](mp[1]) +
                                   gradq[j](mp[2]) * lambda[i](mp[2])};
      AK_(p_idx[i], vx_idx[j]) = AK_(vx_idx[j], p_idx[i]) = gql_ij[0];
      AK_(p_idx[i], vy_idx[j]) = AK_(vy_idx[j], p_idx[i]) = gql_ij[1];
    }
  }
  // Finally multiply with the quadrature weight
  return area / 3.0 * AK_;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_3 */
lf::assemble::COOMatrix<double> buildTaylorHoodGalerkinMatrix(
    const lf::assemble::DofHandler& dofh) {
  // Total number of FE d.o.f.s without Lagrangian multiplier
  lf::assemble::size_type n = dofh.NumDofs();
  // Full Galerkin matrix in triplet format taking into account the zero mean
  // constraint on the pressure.
  lf::assemble::COOMatrix<double> A(n + 1, n + 1);
  // Set up computation of element matrix
  TaylorHoodElementMatrixProvider themp{};
  // Assemble \cor{full} Galerkin matrix for Taylor-Hood FEM
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, themp, A);

  // Add bottom row and right column corresponding to Lagrange multiplier
  // You cannot use AssembleMatrixLocally() because the DofHandler does
  // not know about this extra unknown.
  // Do cell-oriented assembly "manually"
  for (const lf::mesh::Entity* cell : dofh.Mesh()->Entities(0)) {
    LF_ASSERT_MSG(cell->RefEl() == lf::base::RefEl::kTria(),
                  "Only implemented for triangles");
    // Obtain area of triangle
    const double area = lf::geometry::Volume(*cell->Geometry());
    // The pressure GSFs are associated with the nodes
    const std::span<const lf::mesh::Entity* const> nodes{cell->SubEntities(2)};
    // Loop over nodes
    for (const lf::mesh::Entity* node : nodes) {
      // Area of the cell
      // Obtain index of tent function associated with node
      // All indices of global shape functions sitting at node
      std::span<const lf::assemble::gdof_idx_t> dof_idx{
          dofh.InteriorGlobalDofIndices(*node)};
      LF_ASSERT_MSG(dof_idx.size() == 3, "Node must carry 3 dofs!");
      // The index of the tent function is the third one
      const lf::assemble::gdof_idx_t tent_idx = dof_idx[2];
      A.AddToEntry(n, tent_idx, area / 3.0);
      A.AddToEntry(tent_idx, n, area / 3.0);
    }
  }
  // Rely on return value optimization
  return A;
}
/* SAM_LISTING_END_3 */

double allPipeFlow(PowerFlag powerflag, bool producevtk, const char* meshfile,
                   const char* outfile) {
  const std::string& meshfile_str = std::string(meshfile);
  const std::string& outfile_str = std::string(outfile);

  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), meshfile_str);

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh& mesh{*mesh_ptr};

  // Initialize dof handler for the Stokes solver
  lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                         {{lf::base::RefEl::kPoint(), 3},
                                          {lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});

  // Initialize dof handler for the components of the velocity
  lf::assemble::UniformFEDofHandler dofh_u(mesh_ptr,
                                           {{lf::base::RefEl::kPoint(), 1},
                                            {lf::base::RefEl::kSegment(), 1},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});

  // Initialize dof handler for the pressure p
  lf::assemble::UniformFEDofHandler dofh_p(mesh_ptr,
                                           {{lf::base::RefEl::kPoint(), 1},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});

  // We define the right-hand side
  auto g = [](Eigen::VectorXd x) -> Eigen::VectorXd {
    Eigen::VectorXd out(2);
    out(0) = 1.;
    out(1) = 0.;
    return out;
  };

  // Solve the system
  auto res = StokesPipeFlow::solvePipeFlow(dofh, g);

  // Coefficient vectors for the first and second component of the velocity
  Eigen::VectorXd coeff_vec_u1(dofh_u.NumDofs());
  Eigen::VectorXd coeff_vec_u2(dofh_u.NumDofs());

  // Coefficient vector for the pressure
  Eigen::VectorXd coeff_vec_p(dofh_p.NumDofs());

  // We first loop over vertices, then over edges
  for (int codim = 2; codim >= 1; codim--) {
    for (auto e : mesh.Entities(2)) {
      // Global indices for u1, u2 for the respective vertex or edge
      auto glob_idxs = dofh.GlobalDofIndices(*e);
      auto glob_idx_o2 = dofh_u.GlobalDofIndices(*e)[0];

      // Global indices for p for the respective vertex
      lf::assemble::gdof_idx_t glob_idx_o1 = -1;
      if (codim == 2) glob_idx_o1 = dofh_p.GlobalDofIndices(*e)[0];

      // Extract the correct elements for the coefficient matrix of the
      // components of u
      coeff_vec_u1(glob_idx_o2) = res(glob_idxs[0]);
      coeff_vec_u2(glob_idx_o2) = res(glob_idxs[1]);

      // The pressure is only defined on vertices
      if (codim == 2) coeff_vec_p(glob_idx_o1) = res(glob_idxs[2]);
    }
  }

  // Define first- and second-order FE spaces
  auto fes_o1_ptr =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
  auto fes_o2_ptr =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);

  // Define finite-element mesh functions
  lf::fe::MeshFunctionFE<double, double> mf_o2_u1(fes_o2_ptr, coeff_vec_u1);
  lf::fe::MeshFunctionFE<double, double> mf_o2_u2(fes_o2_ptr, coeff_vec_u2);
  lf::fe::MeshFunctionFE<double, double> mf_o1_p(fes_o1_ptr, coeff_vec_p);

  // lf::fe::fe::MeshFUnctionFE<double,double> mf_o2()

  lf::io::VtkWriter vtk_writer(mesh_ptr, outfile_str);

  vtk_writer.WritePointData("u1", mf_o2_u1);
  vtk_writer.WritePointData("u2", mf_o2_u2);
  vtk_writer.WritePointData("p", mf_o1_p);

  return 0;
}

void visualizeTHPipeFlow(const char* meshfile, const char* outfile) {
  (void)allPipeFlow(NOCMOP, true, meshfile, outfile);
}

double computeDissipatedPower(const char* meshfile) {
  return allPipeFlow(VOLUME, false, meshfile);
}

double computeDissipatedPoweBdr(const char* meshfile) {
  return allPipeFlow(BOUNDARY, false, meshfile);
}

}  // namespace StokesPipeFlow
