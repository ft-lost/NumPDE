/**
 * @file asymptoticcvgfem.h
 * @brief NPDE homework AsymptoticCvgFEM code
 * @author R. Hiptmair
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include <cmath>
#include <iostream>

// Lehrfem++ includes
#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/fe/loc_comp_ellbvp.h>
#include <lf/geometry/geometry.h>
#include <lf/io/io.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/fe_space_lagrange_o2.h>
#include <lf/uscalfe/uscalfe.h>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace asymptoticcvgfem {

/** @brief functions computing various errors
 */
/* SAM_LISTING_BEGIN_5 */
template <typename FUNCTOR>
double error_V(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    const Eigen::VectorXd& mu_vec, FUNCTOR sol) {
  const lf::fe::MeshFunctionFE mf_fesol(fes_p, mu_vec);
  const lf::mesh::utils::MeshFunctionGlobal mf_sol{sol};
  return std::sqrt(lf::fe::IntegrateMeshFunction(
      *(fes_p->Mesh()), lf::mesh::utils::squaredNorm(mf_fesol - mf_sol), 6));
}
/* SAM_LISTING_END_5 */
/* SAM_LISTING_BEGIN_2 */
template <typename FUNCTOR>
double error_II(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    const Eigen::VectorXd& mu_vec, FUNCTOR solg) {
  const lf::fe::MeshFunctionGradFE mf_fesolg(fes_p, mu_vec);
  const lf::mesh::utils::MeshFunctionGlobal mf_solg{solg};
  return std::sqrt(lf::fe::IntegrateMeshFunction(
      *(fes_p->Mesh()), lf::mesh::utils::squaredNorm(mf_fesolg - mf_solg), 4));
}
/* SAM_LISTING_END_2 */
/* SAM_LISTING_BEGIN_3 */
template <typename FUNCTOR>
double error_III(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    const Eigen::VectorXd& mu_vec, FUNCTOR sol) {
  const auto bd_flags{
      lf::mesh::utils::flagEntitiesOnBoundary(fes_p->Mesh(), 1)};
  auto bd_sel = [&bd_flags](const lf::mesh::Entity& edge) -> bool {
    return bd_flags(edge);
  };
  const lf::fe::MeshFunctionFE mf_fesol(fes_p, mu_vec);
  const lf::mesh::utils::MeshFunctionGlobal mf_sol{sol};
  return std::sqrt(lf::fe::IntegrateMeshFunction(
      *(fes_p->Mesh()), lf::mesh::utils::squaredNorm(mf_fesol - mf_sol), 6,
      bd_sel, 1));
}
/* SAM_LISTING_END_3 */
/* SAM_LISTING_BEGIN_4 */
template <typename FUNCTOR>
double error_IV(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    const Eigen::VectorXd& mu_vec, FUNCTOR sol) {
  const lf::fe::MeshFunctionFE mf_fesol(fes_p, mu_vec);
  const lf::mesh::utils::MeshFunctionGlobal mf_sol{sol};
  return std::abs(
      lf::fe::IntegrateMeshFunction(*(fes_p->Mesh()),
                                    lf::mesh::utils::squaredNorm(mf_fesol), 6) -
      lf::fe::IntegrateMeshFunction(*(fes_p->Mesh()),
                                    lf::mesh::utils::squaredNorm(mf_sol), 6));
}
/* SAM_LISTING_END_4 */
/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTOR>
double error_I(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    const Eigen::VectorXd& mu_vec, FUNCTOR sol) {
  const lf::fe::MeshFunctionFE mf_fesol(fes_p, mu_vec);
  const lf::mesh::utils::MeshFunctionGlobal mf_sol{sol};
  const auto bd_flags{
      lf::mesh::utils::flagEntitiesOnBoundary(fes_p->Mesh(), 1)};
  auto bd_sel = [&bd_flags](const lf::mesh::Entity& edge) -> bool {
    return bd_flags(edge);
  };
  return std::abs(lf::fe::IntegrateMeshFunction(
      *(fes_p->Mesh()), mf_fesol - mf_sol, 4, bd_sel, 1));
}
/* SAM_LISTING_END_1 */

/** @brief Framework for convergence study
 *
 * Convergence studied on sequence of regularly refined meshes of a triangular
 * domain
 */
Eigen::MatrixXd studyAsymptoticCvg(unsigned int reflev = 4);

/** @brief Compute finite element solution of impedance problem for Laplacian

    @return basis expansion vector of FE solution
 */
/* SAM_LISTING_BEGIN_9 */
template <typename SOURCE_FUNCTOR, typename IMPDATA_FUNCTOR>
Eigen::VectorXd solveLaplImpBVP(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_p,
    SOURCE_FUNCTOR f, IMPDATA_FUNCTOR h) {
  // Set up mesh functions for data
  lf::mesh::utils::MeshFunctionGlobal mf_f{f};
  lf::mesh::utils::MeshFunctionGlobal mf_h{h};
  lf::mesh::utils::MeshFunctionGlobal mf_one{
      [](Eigen::VectorXd /*x*/) -> double { return 1.0; }};

  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler& dofh{fes_p->LocGlobMap()};
  // Dimension of finite element space
  const std::size_t N_dofs(dofh.NumDofs());

  // I. Assembly of Galerkin matrix
  // Set up element matrix providers
  // For Dirichlet form in the volume
  lf::fe::DiffusionElementMatrixProvider<double, decltype(mf_one)>
      vol_elmat_provider(fes_p, mf_one);
  // For boundary contribution to the bilinear form
  // Obtain an array of boolean flags for the edges of the mesh, 'true'
  // indicates that the edge lies on the boundary
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fes_p->Mesh(), 1)};
  // Selector for edges on the boundary
  auto imp_sel = [&bd_flags](const lf::mesh::Entity& edge) -> bool {
    return bd_flags(edge);
  };
  lf::fe::MassEdgeMatrixProvider<double, decltype(mf_one), decltype(imp_sel)>
      bd_elmat_provider(fes_p, mf_one, imp_sel);
  // Assemble Galerkin matrix
  // Matrix in triplet format holding Galerkin matrix, zero initially.
  lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);
  // Cell-oriented assembly
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, vol_elmat_provider, A);
  // Edge-oriented assembly of boundary term
  lf::assemble::AssembleMatrixLocally(1, dofh, dofh, bd_elmat_provider, A);

  // II. Assembly of right-hand side vector
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs);
  phi.setZero();
  // Assemble volume part of right-hand side vector depending on the source
  // function f.
  lf::fe::ScalarLoadElementVectorProvider<double, decltype(mf_f)> elvec_builder(
      fes_p, mf_f);
  // Invoke assembly on cells (codim == 0)
  lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, phi);
  // Supply boundary contributions due to impedance boundary conditions
  lf::fe::ScalarLoadEdgeVectorProvider<double, decltype(mf_h),
                                       decltype(imp_sel)>
      bd_elvec_provider(fes_p, mf_h, imp_sel);
  // Invoke assembly on edges (codim == 1), update vector
  lf::assemble::AssembleVectorLocally(1, dofh, bd_elvec_provider, phi);

  // III. Solve linear system
  // Assembly completed: Convert COO matrix A into CRS format using Eigen's
  // internal conversion routines.
  Eigen::SparseMatrix<double> A_crs = A.makeSparse();

  // Solve linear system using Eigen's sparse direct elimination
  // Examine return status of solver in case the matrix is singular
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "LU decomposition failed");
  const Eigen::VectorXd sol_vec = solver.solve(phi);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "Solving LSE failed");
  return sol_vec;
}
/* SAM_LISTING_END_9 */

}  // namespace asymptoticcvgfem
