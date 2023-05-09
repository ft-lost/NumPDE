/**
 * @file transpsemilagr.h
 * @brief NPDE homework TranspSemiLagr code
 * @author Philippe Peter
 * @date November 2020
 * @copyright Developed at SAM, ETH Zurich
 */
#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <memory>

#include "local_assembly.h"

namespace TranspSemiLagr {

/**
 * @brief encorce zero dirichlet boundary conditions
 * @param fe_space shared point to the fe spce on which the problem is defined.
 * @param A reference to the square coefficient matrix in COO format
 * @param b reference to the right-hand-side vector
 */
void enforce_zero_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b);

/**
 * @brief performs a semi lagrangian step according to the update
 * formula 7.3.4.13
 * @param fe_space finite element space on which the problem is solved
 * @param u0_vector nodal values of the solution at the previous time step
 * @param v velocity field (time independent)
 * @param tau time step size
 * @return nodal values of the approximated solution at current time step
 */
template <typename FUNCTOR>
class SemiLagrStep {
 public:
  SemiLagrStep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
      FUNCTOR v)
      : fe_space_(fe_space),
        v_(v),
        A_lm_(fe_space->LocGlobMap().NumDofs(),
              fe_space->LocGlobMap().NumDofs()) {
    // lumped mass matrix A_lm
    LumpedMassElementMatrixProvider lumped_mass_element_matrix_provider(
        [](Eigen::Vector2d /*x*/) { return 1.0; });
    lf::assemble::AssembleMatrixLocally(
        0, fe_space->LocGlobMap(), fe_space->LocGlobMap(),
        lumped_mass_element_matrix_provider, A_lm_);
  };

  Eigen::VectorXd step(const Eigen::VectorXd& u0_vector, double tau) {
    // Assemble left hand side A = A_lm + tau*A_s
    // stiffness matrix tau*A_s
    lf::assemble::COOMatrix<double> A = A_lm_;
    //====================
    // Your code goes here
    //====================
    return Eigen::VectorXd::Ones(u0_vector.size());
  }

 private:
  std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
  FUNCTOR v_;
  lf::assemble::COOMatrix<double> A_lm_;
};

/**
 * @brief approximates the solution to the first model problem specified in the
 * exercise sheet based on N uniform timesteps of the Semi Lagrangian method
 * @param fe_space (linear) finite element space on which the solution is
 * approximated
 * @param u0_vector vector of nodal values of the initial condition
 * @param N number of time steps
 * @param T final time
 */
Eigen::VectorXd solverot(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd u0_vector, int N, double T);

/**
 * @param solves the variational evolution problem specified in the exercise
 * sheet over on time step.
 * @param fe_space finite element space on which the variational evolution
 * problem is solved
 * @param u0_vector nodal values of the solution at the previous time step
 * @param c coefficient function in the variational evolution problem
 * @param tau time step size
 */
/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTOR>
class ReactionStep {
 public:
  ReactionStep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
      FUNCTOR c)
      : fe_space_(fe_space), c_(c) {
    //====================
    // Your code goes here
    //====================
  };

  Eigen::VectorXd step(const Eigen::VectorXd& u0_vector, double tau) {
    //====================
    // Your code goes here
    //====================
    return Eigen::VectorXd::Ones(u0_vector.size());
  }

 private:
  std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
  FUNCTOR c_;
  Eigen::SparseMatrix<double> mass_matrix_c_sparse_;
  Eigen::SparseMatrix<double> mass_matrix_1_sparse_;
};
/* SAM_LISTING_END_1 */

/**
 * @brief approximates the solution to the second model problem specified in the
 * exercise sheet based on N uniform time steps of the Strang-splitting
 * split-step method
 * @param fe_space (linear) finite element space on which the solution is
 * approximated
 * @param u0_vector vector of nodal values of the initial condition
 * @param N number of time steps
 * @param T final time
 */
Eigen::VectorXd solvetrp(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd u0_vector, int N, double T);

}  // namespace TranspSemiLagr
