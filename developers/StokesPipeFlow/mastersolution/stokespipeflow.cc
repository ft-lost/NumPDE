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
#include <lf/mesh/test_utils/test_meshes.h>

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
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
#if SOLUTION
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
  AK_ *= area / 3.0;
#else
/* **********************************************************************
   Your code here
   ********************************************************************** */
#endif
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Workaround for a side-effect in LehrFEM++
  // Local shape functions associated with edges have to be swapped, if the edge
  // has "negative" orientation with respect to the triangle. This is meant to
  // offset such a swapping done by LehrFEM++'s UniformFEDofHandler.
  auto edge_orientations = cell.RelativeOrientations();
  LF_ASSERT_MSG(edge_orientations.size() == 3,
                "Triangle should have 3 edges!?");
  Eigen::VectorXd tmp(15);
  for (int k = 0; k < 3; ++k) {
    const lf::assemble::ldof_idx_t ed_dofx = 9 + 2 * k;
    const lf::assemble::ldof_idx_t ed_dofy = 10 + 2 * k;
    if (edge_orientations[k] == lf::mesh::Orientation::negative) {
      // The rows and columns of the element matrix with numbers 9+2*k and
      // 9+2*k+1 have to be swapped.
      tmp = AK_.col(ed_dofx);
      AK_.col(ed_dofx) = AK_.col(ed_dofy);
      AK_.col(ed_dofy) = tmp;
      tmp = AK_.row(ed_dofx);
      AK_.row(ed_dofx) = AK_.row(ed_dofy);
      AK_.row(ed_dofy) = tmp;
    }
  }
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Finally multiply with the quadrature weight
  return AK_;
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
#if SOLUTION
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
#else
/* **********************************************************************
   Your code here
   ********************************************************************** */
#endif
  // Rely on return value optimization
  return A;
}
/* SAM_LISTING_END_3 */

double allPipeFlow(PowerFlag powerflag, bool producevtk, const char* meshfile,
                   const char* outfile) {
  LF_VERIFY_MSG((meshfile != nullptr), "Must provide a mesh file");
  const std::string meshfile_str = std::string(meshfile);

  /* SAM_LISTING_BEGIN_5 */
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), meshfile_str);

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh& mesh{*mesh_ptr};

  // Initialize dof handler for Taylor-Hood FEM
  lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                         {{lf::base::RefEl::kPoint(), 3},
                                          {lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});
// We define function providing the boundary values for the velocity
#if SOLUTION
  auto g = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    if (x[0] < 1E-8) {
      // Left boundary: inlet, parabolic velocity profile
      return {(1.0 - x[1]) * (x[1] - 0.5), 0.0};
    }
    if (x[0] > 0.99999) {
      // Right boundary: outlet
      return {(0.5 - x[1]) * x[1], 0.0};
    }
    return {0.0, 0.0};
  };
#else
  /* **********************************************************************
     Replace this placeholder code
     ********************************************************************** */
  auto g = [&vdir](Eigen::Vector2d /*x*/) -> Eigen::Vector2d {
    return {0.0, 0.0};
  };
#endif
  // Solve the system
  auto res = StokesPipeFlow::solvePipeFlow(dofh, g);

  if (producevtk) {
    LF_VERIFY_MSG(outfile != nullptr,
                  "Filename for .vtk files has to be provided");
    // Define first- and second-order Lagrangian FE spaces for the piecewise
    // linear Taylor-Hood pressure approximation and the piecewise quadratic
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    auto fes_o2_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o2_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure p
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();

    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p(dofh_p.NumDofs());

    // Loop over vertices and edges to remap the basis expansion coefficients
    for (int codim = 2; codim >= 1; codim--) {
      for (auto e : mesh.Entities(codim)) {
        // Global indices for u1, u2 for the respective vertex or edge
        auto glob_idxs = dofh.GlobalDofIndices(*e);
        auto glob_idx_o2 = dofh_u.GlobalDofIndices(*e)[0];

        // Extract the correct elements for the coefficient matrix of the
        // components of u
        coeff_vec_u1(glob_idx_o2) = res(glob_idxs[0]);
        coeff_vec_u2(glob_idx_o2) = res(glob_idxs[1]);

        // Global indices for p for the respective vertex
        lf::assemble::gdof_idx_t glob_idx_o1 = -1;
        // The pressure is only defined on vertices
        if (codim == 2) {
          glob_idx_o1 = dofh_p.GlobalDofIndices(*e)[0];
          coeff_vec_p(glob_idx_o1) = res(glob_idxs[2]);
        }
      }
    }

    // Define finite-element mesh functions
    lf::fe::MeshFunctionFE<double, double> mf_o2_u1(fes_o2_ptr, coeff_vec_u1);
    lf::fe::MeshFunctionFE<double, double> mf_o2_u2(fes_o2_ptr, coeff_vec_u2);
    lf::fe::MeshFunctionFE<double, double> mf_o1_p(fes_o1_ptr, coeff_vec_p);

    const std::string outfile_str = std::string(outfile);
    lf::io::VtkWriter vtk_writer(mesh_ptr, outfile_str);
    vtk_writer.WritePointData("u1", mf_o2_u1);
    vtk_writer.WritePointData("u2", mf_o2_u2);
    vtk_writer.WritePointData("p", mf_o1_p);
  }
  /* SAM_LISTING_END_5 */
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

void testCvgTaylorHood(unsigned int refsteps) {
  // Analytic solution for velocity and pressure
  const Eigen::Vector2d d{1.0, 2.0};
  auto v_ex = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double nx = x.norm();
    return 0.25 * (-std::log(nx) * d + x * (x.dot(d)) / (nx * nx));
  };
  auto p_ex = [&d](Eigen::Vector2d x) -> double {
    x += Eigen::Vector2d(0.5, 0.5);
    return -0.5 * x.dot(d) / x.squaredNorm();
  };

  // Generate a small unstructured triangular mesh
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_ptr,
                                                              refsteps);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();

  // Retrieve meshes on all levels
  for (int level = 0; level < L; ++level) {
    const std::shared_ptr<const lf::mesh::Mesh> lev_mesh_p =
        multi_mesh.getMesh(level);
    // Initialize dof handler for Taylor-Hood FEM
    lf::assemble::UniformFEDofHandler dofh(lev_mesh_p,
                                           {{lf::base::RefEl::kPoint(), 3},
                                            {lf::base::RefEl::kSegment(), 2},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
    // Define first- and second-order Lagrangian FE spaces for the piecewise
    // linear Taylor-Hood pressure approximation and the piecewise quadratic
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(lev_mesh_p);
    auto fes_o2_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(lev_mesh_p);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o2_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure p
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();
    // Solve the system with trace of the exact velocity solution as Dirichlet
    // data
    auto res = StokesPipeFlow::solvePipeFlow(dofh, v_ex);
    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p(dofh_p.NumDofs());

    // Remapping dofs: We first loop over vertices, then over edges
    for (int codim = 2; codim >= 1; codim--) {
      for (auto e : lev_mesh_p->Entities(codim)) {
        // Global indices for u1, u2 for the respective vertex or edge
        auto glob_idxs = dofh.GlobalDofIndices(*e);
        auto glob_idx_o2 = dofh_u.GlobalDofIndices(*e)[0];
        // Extract the correct elements for the coefficient matrix of the
        // components of u
        coeff_vec_u1(glob_idx_o2) = res(glob_idxs[0]);
        coeff_vec_u2(glob_idx_o2) = res(glob_idxs[1]);
        // The pressure is only defined on vertices
        if (codim == 2) {
          // Global indices for p for the respective vertex
          lf::assemble::gdof_idx_t glob_idx_o1 = dofh_p.GlobalDofIndices(*e)[0];
          coeff_vec_p(glob_idx_o1) = res(glob_idxs[2]);
        }
      }
    }
    // Define finite-element mesh functions
    lf::fe::MeshFunctionFE<double, double> mf_o2_u1(fes_o2_ptr, coeff_vec_u1);
    lf::fe::MeshFunctionFE<double, double> mf_o2_u2(fes_o2_ptr, coeff_vec_u2);
    lf::fe::MeshFunctionFE<double, double> mf_o1_p(fes_o1_ptr, coeff_vec_p);

    // Exact solution for the first component of the velocity
    auto u1 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[0]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u1{u1};
    // Exact solution second component of  the velocity
    auto u2 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[1]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u2{u2};
    // Mesh funcvtion for exact solution pressure
    const lf::mesh::utils::MeshFunctionGlobal mf_p{p_ex};
    // compute errors with 5th order quadrature rules
    double L2err_u1 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u1 - mf_u1), 4));
    double L2err_u2 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u2 - mf_u2), 4));
    double L2err_p = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o1_p - mf_p), 4));
    std::cout << "Level " << level << ": L2 Error u1: " << L2err_u1
              << ", L2 Error u2: " << L2err_u2 << ", L2 Error p: " << L2err_p
              << std::endl;
  }
}

}  // namespace StokesPipeFlow
