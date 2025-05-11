/**
 * @file gausslobattoparabolic.h
 * @brief NPDE exam TEMPLATE CODE FILE
 * @author Oliver Rietmann
 * @date 22.07.2020
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef GAUSSLOBATTOPARABOLIC_H_
#define GAUSSLOBATTOPARABOLIC_H_

#include <lf/assemble/assemble.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <functional>

namespace GaussLobattoParabolic {

/**
 * @brief Compute the matrix \Blue{$\tilde{M}$} described in the exercise.
 *
 * @param fe_space finite element space, say of dimension \Blue{$N$}.
 * @return matrix of size \Blue{$N\times N$}
 */
lf::assemble::COOMatrix<double> initMbig(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space);

/**
 * @brief Compute the matrix \Blue{$\tilde{A}$} described in the exercise.
 *
 * @param fe_space finite element space, say of dimension \Blue{$N$}.
 * @return matrix of size \Blue{$N\times N$}
 */
lf::assemble::COOMatrix<double> initAbig(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space);

/* SAM_LISTING_BEGIN_3 */
class RHSProvider {
 public:
  // Disabled constructors
  RHSProvider() = delete;
  RHSProvider(const RHSProvider &) = delete;
  RHSProvider(RHSProvider &&) = delete;
  RHSProvider &operator=(const RHSProvider &) = delete;
  RHSProvider &operator=(const RHSProvider &&) = delete;
  // Main constructor; precomputations to be done here
  RHSProvider(const lf::assemble::DofHandler &dofh,
              std::function<double(double)> g);
  // Destructor
  virtual ~RHSProvider() = default;
  /* Class member functions */
  // Evaluation operator for right-hand-side vector
  Eigen::VectorXd operator()(double t) const;

 private:
  std::function<double(double)> g_;
  //====================
  //
  Eigen::VectorXd zero_one;
  //====================
};
/* SAM_LISTING_END_3 */

/**
 * @brief Compute the expansion coefficient vector \Blue{$\mu$} of the
 * solution at time T, corresponding to the basis of fe_space,
 * with vanishing inital data.
 *
 * @param fe_space finite element space, say of dimension \Blue{$N$}.
 * @param T final time
 * @param M number of time steps for the Gauss-Lobatto IIIC scheme
 * @param g continuously differentiable function satisfying \Blue{$g(0)=0$} that
 * describes the time-dependent boundary conditions.
 * @param u0 inital data satisfying \Blue{$g(0)=u_0$} on the boundary
 * @return vector of size \Blue{$N$} holding the coefficients of the solution
 */
/* SAM_LISTING_BEGIN_4 */
template <typename GFUNCTION>
Eigen::VectorXd evolveIBVPGaussLobatto(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    double T, unsigned int M, GFUNCTION &&g) {
  // timestep size
  const double tau = T / M;

  const lf::assemble::DofHandler &dofh = fe_space->LocGlobMap();
  const int N = dofh.NumDofs();

  // Coefficient vector, initial value = 0
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(N);

  // Build left-hand side sparse block matrix

  //====================
  lf::assemble::COOMatrix<double> lhs(2 * N, 2 * N);
  // Arrange blocks by triplet manipulations
  // First: Two copies of \tilde{M} on the diagonal
  lf::assemble::COOMatrix<double> COO_M = initMbig(fe_space);
  for (const Eigen::Triplet<double> &triplet : COO_M.triplets()) {
    lhs.AddToEntry(triplet.row(), triplet.col(), triplet.value());
    lhs.AddToEntry(triplet.row() + N, triplet.col() + N, triplet.value());
  }
  // Scaled copies of \tilde{A} added to diagonal and set as off-diagonal blocks
  lf::assemble::COOMatrix<double> COO_A = initAbig(fe_space);
  for (const Eigen::Triplet<double> &triplet : COO_A.triplets()) {
    const int row = triplet.row();
    const int col = triplet.col();
    const double value = 0.5 * tau * triplet.value();
    lhs.AddToEntry(row, col, value);
    lhs.AddToEntry(row, col + N, -value);
    lhs.AddToEntry(row + N, col, value);
    lhs.AddToEntry(row + N, col + N, value);
  }
  Eigen::SparseMatrix<double> A = COO_A.makeSparse();
  // Then ompute the LU decomposition outside of the timestepping loop:
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(lhs.makeSparse());
  // Helper class for computuing source term
  RHSProvider rhs_provider(dofh, std::move(g));


  for(int i = 0; i < M; ++i){
    Eigen::VectorXd phi1 = rhs_provider(i*tau);
    Eigen::VectorXd phi2 = rhs_provider((i+1)*tau);
    Eigen::VectorXd rhs(2*N);
    rhs << phi1 - A*mu, phi2 - A*mu;
    Eigen::VectorXd k = solver.solve(rhs);
    mu = mu + 0.5*tau*(k.head(N) + k.tail(N));
  }
  // ...
  // Perform Timestepping here relying on the evaluation operators of
  // RHSProvider to obtain the time-dependent right-hand-side vector in
  // each timestep
  // ...
  //====================
  return mu;
}
/* SAM_LISTING_END_4 */

}  // namespace GaussLobattoParabolic

#endif  // #define GAUSSLOBATTOPARABOLIC_H_
