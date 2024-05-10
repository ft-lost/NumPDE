#ifndef IMEX_H
#define IMEX_H
/**
 * @file imexrkssm.h
 * @brief NPDE homework IMEXRKSSM code
 * @author Bob Schreiner
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <memory>
#include <utility>

namespace IMEX {

/**
 *
 * @param M
 * @return the error given
 */
double IMEXError(int M);

/**
 * @brief Assembly of general Galerkin matrix for 2nd-order elliptic
 *        boundary value problem
 *
 * @tparam FUNC_ALPHA functor type for diffusion coefficient
 * @tparam FUNC_GAMMA functor type for reaction coefficient
 * @tparam FUNC_BETA functor type for impedance coefficient
 *
 * @param dofh DofHandler object providing mesh a local-to-global
 *        index mapping for global shape functions.
 *
 * This function computes the finite element Galerkin matrix for the
 * lowest-order Lagrangian FEM and the bilinear form on \f$H^1(\Omega)\f$
 * \f[
     (u,v) \mapsto \int\limits_{\Omega} \alpha(\mathbf{x})
        \mathbf{grad}\,u\cdot\mathfb{grad}\,v +
        \gamma(\mathbf{x})\,u\,v\,\mathrm{d}\mathbf{x} +
        \int\limits_{\Omega}\beta(\mathbf{x})\,u\,v\,\mathrm{d}S(\mathbf{x})
    \f]
 */
template <typename FUNC_ALPHA, typename FUNC_GAMMA, typename FUNC_BETA>
Eigen::SparseMatrix<double> compGalerkinMatrix(
    const lf::assemble::DofHandler& dofh, FUNC_ALPHA&& alpha,
    FUNC_GAMMA&& gamma, FUNC_BETA&& beta) {
  // obtain mesh and set up fe_space (p.w. linear Lagrangian FEM)
  auto mesh = dofh.Mesh();
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

  // get the number of degrees of freedom = dimension of FE space
  const lf::base::size_type N_dofs(dofh.NumDofs());
  // Set up an empty sparse matrix to hold the Galerkin matrix
  lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);
  // Initialize ELEMENT_MATRIX_PROVIDER object
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};
  lf::mesh::utils::MeshFunctionGlobal mf_gamma{gamma};
  lf::uscalfe::ReactionDiffusionElementMatrixProvider elmat_builder(
      fe_space, std::move(mf_alpha), std::move(mf_gamma));
  // Cell-oriented assembly over the whole computational domain
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

  // Add contributions of boundary term in the bilinear form using
  // a LehrFEM++ built-in high-level ENTITY_MATRIX_PROVIDER class
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};
  lf::mesh::utils::MeshFunctionGlobal mf_beta{beta};
  lf::uscalfe::MassEdgeMatrixProvider edgemat_builder(
      fe_space, std::move(mf_beta), bd_flags);
  lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edgemat_builder, A);
  return A.makeSparse();
}

Eigen::VectorXd compNonlinearTerm(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    const Eigen::VectorXd& u);

/* SAM_LISTING_BEGIN_1 */
struct AssemblerLocalFunc {
  template <typename BUILDER_DATA, typename ELEMENT>
  inline static Eigen::Vector3d Eval(const BUILDER_DATA& data,
                                     const ELEMENT& el) {
#if SOLUTION
    Eigen::Vector3d res;
    for (int i = 0; i < 3; ++i) {
      res[i] = std::sinh(data.u_loc[i]);
    }
    const double vol = lf::geometry::Volume(*el.Geometry());
    return res * vol / 3.;
#else
    Eigen::Vector3d res = Eigen::Vector3d::Zero();
    // ========================================
    // Your code here
    // ========================================
    return res;
#endif
  };
};
/* SAM_LISTING_END_1 */

class IMEXTimestep {
 public:
  IMEXTimestep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test);
  void compTimestep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
      double tau, Eigen::VectorXd& y) const;

 private:
  Eigen::SparseMatrix<double> M_;
  Eigen::SparseMatrix<double> A_;
  Eigen::VectorXd phi_;
  // Feel free to add more data members

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_M_;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_A_;
  // Very bad: huge dense matrices! Must be avoided!
  Eigen::MatrixXd MInvA_;
  Eigen::VectorXd MInvphi_;
};

Eigen::VectorXd solveTestProblem(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    int M);

void visSolution(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd& u, const std::string&& filename);

}  // namespace IMEX

#endif  // IMEX_H
