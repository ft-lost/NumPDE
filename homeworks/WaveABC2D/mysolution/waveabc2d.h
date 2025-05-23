#ifndef WAVEABC2D_HPP
#define WAVEABC2D_HPP

/** @file
 * @brief NPDE WaveABC2D
 * @author Erick Schulz
 * @date 11/12/2019
 * @copyright Developed at ETH Zurich
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Lehrfem++ includes
#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

namespace WaveABC2D {

Eigen::VectorXd scalarImplicitTimestepping(double epsilon, unsigned int M);

void testConvergenceScalarImplicitTimestepping();

template <typename FUNC_ALPHA, typename FUNC_BETA, typename FUNC_GAMMA>
lf::assemble::COOMatrix<double> computeGalerkinMat(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p,
    FUNC_ALPHA alpha, FUNC_GAMMA gamma, FUNC_BETA beta) {
  // Pointer to current mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = fe_space_p->Mesh();
  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler &dofh{fe_space_p->LocGlobMap()};
  // Dimension of finite element space
  const lf::uscalfe::size_type N_dofs(dofh.NumDofs());

  /* Creating coefficient-functions as Lehrfem++ mesh functions */
  // Coefficient-functions used in the class template
  // ReactionDiffusionElementMatrixProvider and MassEdgeMatrixProvider
  auto alpha_mf = lf::mesh::utils::MeshFunctionGlobal(alpha);
  auto gamma_mf = lf::mesh::utils::MeshFunctionGlobal(gamma);
  auto beta_mf = lf::mesh::utils::MeshFunctionGlobal(beta);

  // Instantiating Galerkin matrix to computed
  // This matrix is in triplet format, zero initially.
  lf::assemble::COOMatrix<double> galMat_COO(N_dofs, N_dofs);

  /* Initialization of local matrices builders */
  // Initialize objects taking care of local computations for volume integrals
  lf::uscalfe::ReactionDiffusionElementMatrixProvider<
      double, decltype(alpha_mf), decltype(gamma_mf)>
      elem_builder(fe_space_p, alpha_mf, gamma_mf);
  // Initialize objects taking care of local computations for boundary integrals
  // Creating a predicate that will guarantee that the computations for the
  // boundary mass matrix are carried only on the edges of the mesh
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
  lf::uscalfe::MassEdgeMatrixProvider<double, decltype(beta_mf),
                                      decltype(bd_flags)>
      bd_mat_builder(fe_space_p, beta_mf, bd_flags);

  /* Assembling the Galerkin matrices */
  // Information about the mesh and the local-to-global map is passed through
  // a Dofhandler object, argument 'dofh'. This function call adds triplets to
  // the internal COO-format representation of the sparse matrix.
  // Invoke assembly on cells (co-dimension = 0 as first argument)
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elem_builder, galMat_COO);
  // Invoke assembly on edges (co-dimension = 1 as first argument)
  lf::assemble::AssembleMatrixLocally(1, dofh, dofh, bd_mat_builder,
                                      galMat_COO);

  return galMat_COO;
}

class progress_bar {
  static const auto overhead = sizeof " [100%]";
  std::ostream &os;
  const std::size_t bar_width;
  std::string message;
  const std::string full_bar;

 public:
  progress_bar(std::ostream &os, std::size_t line_width, std::string message_,
               const char symbol = '.')
      : os{os},
        bar_width{line_width - overhead},
        message{std::move(message_)},
        full_bar{std::string(bar_width, symbol) + std::string(bar_width, ' ')} {
    if (message.size() + 1 >= bar_width ||
        message.find('\n') != std::string::npos) {
      os << message << '\n';
      message.clear();
    } else {
      message += ' ';
    }
    write(0.0);
  }

  progress_bar(const progress_bar &) = delete;
  progress_bar &operator=(const progress_bar &) = delete;

  ~progress_bar() {
    write(1.0);
    os << '\n';
  }

  void write(double fraction);
};  // class progress_bar

/** @brief class providing timestepping for WaveABC2D */
/* SAM_LISTING_BEGIN_9 */
template <typename FUNC_RHO, typename FUNC_MU0, typename FUNC_NU0>
class WaveABC2DTimestepper {
 public:
  // Main constructor; precomputations are done here
  WaveABC2DTimestepper(
      const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p,
      FUNC_RHO rho, unsigned int M, double T);

  // Public member functions
  Eigen::VectorXd solveWaveABC2D(FUNC_MU0 mu0, FUNC_NU0 nu0);
  double energies();

 private:
  double T_;          // final time
  unsigned int M_;    // nb of steps
  double step_size_;  // time inverval
  std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_p_;
  bool timestepping_performed_;  // bool to assert that energies are computed
//====================
// Your code goes here
//====================
};  // class WaveABC2DTimestepper
/* SAM_LISTING_END_9 */

/* Implementing constructor of class WaveABC2DTimestepper */
/* SAM_LISTING_BEGIN_1 */
template <typename FUNC_RHO, typename FUNC_MU0, typename FUNC_NU0>
WaveABC2DTimestepper<FUNC_RHO, FUNC_MU0, FUNC_NU0>::WaveABC2DTimestepper(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p,
    FUNC_RHO rho, unsigned int M, double T)

    : fe_space_p_(fe_space_p),
      M_(M),
      T_(T),
      step_size_(T / M),
      timestepping_performed_(false) {
  /* Creating coefficient-functions as Lehrfem++ mesh functions */
  // Coefficient-functions used in the class template
  // ReactionDiffusionElementMatrixProvider and MassEdgeMatrixProvider
  auto rho_mf = lf::mesh::utils::MeshFunctionGlobal(rho);
  auto zero_mf = lf::mesh::utils::MeshFunctionGlobal(
      [](Eigen::Vector2d) -> double { return 0.0; });
  auto one_mf = lf::mesh::utils::MeshFunctionGlobal(
      [](Eigen::Vector2d) -> double { return 1.0; });

//====================
// Your code goes here
//====================
}
/* SAM_LISTING_END_1 */

/* Implementing member functions of class WaveABC2DTimestepper */
/* SAM_LISTING_BEGIN_2 */
template <typename FUNC_RHO, typename FUNC_MU0, typename FUNC_NU0>
Eigen::VectorXd WaveABC2DTimestepper<FUNC_RHO, FUNC_MU0,
                                     FUNC_NU0>::solveWaveABC2D(FUNC_MU0 mu0,
                                                               FUNC_NU0 nu0) {
  std::cout << "\nSolving variational problem of WaveABC2D." << std::endl;
  Eigen::VectorXd sol;

  // Initial conditions
  auto mf_mu0 = lf::mesh::utils::MeshFunctionGlobal(mu0);
  auto mf_nu0 = lf::mesh::utils::MeshFunctionGlobal(nu0);
  Eigen::VectorXd nu0_nodal = lf::fe::NodalProjection(*fe_space_p_, mf_nu0);
  Eigen::VectorXd mu0_nodal = lf::fe::NodalProjection(*fe_space_p_, mf_mu0);

//====================
// Your code goes here
//====================

  return sol;
}  // solveWaveABC2D
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_10 */
template <typename FUNC_RHO, typename FUNC_MU0, typename FUNC_NU0>
double WaveABC2DTimestepper<FUNC_RHO, FUNC_MU0, FUNC_NU0>::energies() {
  double energy;
//====================
// Your code goes here
//====================
  return energy;
}
/* SAM_LISTING_END_10 */

}  // namespace WaveABC2D

#endif
