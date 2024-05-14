/**
 * @file imexrkssm.cc
 * @brief NPDE homework IMEXRKSSM code
 * @author Bob Schreiner
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "imexrkssm.h"

#include <lf/assemble/dofhandler.h>
#include <lf/mesh/utils/mesh_data_set.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <utility>

namespace IMEX {

/* SAM_LISTING_BEGIN_1 */
double IMEXError(int M) {
  const double tau = 1. / M;
  // We store the error for each intermediate step we take
  std::vector<double> err(M + 1);
  err[0] = 0;
  // We store each intermediate step we take
  std::vector<double> y_rk(M + 1);
  // Initialize $y_0$ with 1/4
  y_rk[0] = 0.25;
  // Compute gamma given by the RKS
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;

  // Store the increment vectors
  Eigen::Vector2d k;
  Eigen::Vector3d k_hat;

  // Store the Butcher Matrices in Eigen::Matrices for ease of use
  Eigen::Matrix2d a;
  a << gamma, 0.0, 1.0 - 2.0 * gamma, gamma;
  const Eigen::Vector2d b{0.5, 0.5};
  Eigen::Matrix3d a_hat;
  a_hat << 0., 0., 0., gamma, 0., 0., gamma - 1., 2.0 * (1.0 - gamma), 0.;
  const Eigen::Vector3d b_hat{0.0, 0.5, 0.5};

  // ========================================
  // Your code here
  // ========================================
  // Return the maximal error we made during the time stepping
  return *std::max_element(err.begin(), err.end());
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
Eigen::VectorXd compNonlinearTerm(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    const Eigen::VectorXd& u) {
  // Get the DofHandler
  const lf::assemble::DofHandler& dofh = fe_test->LocGlobMap();
  // Get the mesh
  const std::shared_ptr<const lf::mesh::Mesh> mesh_p = fe_test->Mesh();
  // Initialize the vector that will hold the nonLinearTerm
  Eigen::VectorXd nonLinearTerm = Eigen::VectorXd::Zero(dofh.NumDofs());
  // Define the data structure , that will be passed on to the
  // localVectorAssembler
  struct data_t {
    Eigen::VectorXd u;
    Eigen::Vector3d u_loc;
    explicit data_t(Eigen::VectorXd u) : u(std::move(u)){};
  } data(u);

  // ========================================
  // Your code here
  // ========================================
  // Return the entire nonLinearTerm
  return nonLinearTerm;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
IMEXTimestep::IMEXTimestep(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    const double tau, const Eigen::VectorXd& butcher_matrix_diag)
    : tau_(tau) {
  // ========================================
  // Your code here
  // Define some helper functions that we can pass to compGalerkinMatrix as
  // alpha, beta and gamma

  // Compute the rhs vector

  // Compute the LHS time independent Matrices

  // Do some pre-computations to initialize the sparse solvers

  // ========================================
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
void IMEXTimestep::compTimestep(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    Eigen::VectorXd& y) const {
  // Define gamma
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;
  const int N = fe_test->LocGlobMap().NumDofs();

  // Define the increment Vectors and store the in the columns of a
  // Eigen::Matrix
  Eigen::MatrixXd kappa(N, 2);
  Eigen::MatrixXd kappa_hat(N, 3);

  // Define the butcher Matrices
  Eigen::Matrix2d a;
  a << gamma, 0.0, 1.0 - 2.0 * gamma, gamma;
  const Eigen::Vector2d b{0.5, 0.5};
  Eigen::Matrix3d a_hat;
  a_hat << 0., 0., 0., gamma, 0., 0., gamma - 1., 2.0 * (1.0 - gamma), 0.;
  const Eigen::Vector3d b_hat{0.0, 0.5, 0.5};

  // ========================================
  // Your code here
  // ========================================
}
/* SAM_LISTING_END_4 */

Eigen::VectorXd solveTestProblem(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    int M) {
  const unsigned int N = fe_space->LocGlobMap().NumDofs();
  const double tau = 1. / M;
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;
  // Define the butcher Matrix a
  Eigen::Matrix2d a;
  a << gamma, 0.0, 1.0 - 2.0 * gamma, gamma;

  // Create the timestepper
  IMEXTimestep Timestepper(fe_space, tau, a.diagonal());

  // Initialize $u_0 = sin(x*2pi)*sin(y*2pi)+1$
  Eigen::VectorXd u = Eigen::VectorXd::Zero(N);

  auto initial_values_sin = [](const Eigen::Vector2d& x) {
    return std::sin(x(0) * 2.0 * M_PI) * std::sin(x(1) * 2.0 * M_PI) + 1.0;
  };

  lf::mesh::utils::MeshFunctionGlobal mf_init(initial_values_sin);
  lf::fe::ScalarLoadElementVectorProvider element_vector_provider(fe_space,
                                                                  mf_init);
  lf::assemble::AssembleVectorLocally(0, fe_space->LocGlobMap(),
                                      element_vector_provider, u);

  for (unsigned int i = 0; i < M; ++i) {
    Timestepper.compTimestep(fe_space, u);
    // Uncomment the following lines to get a visualization of the whole
    // timeseries

    // std::stringstream step_string;
    // step_string << "solution_step_" << i << ".vtk";
    // visSolution(fe_space, u, step_string.str());
  }
  return u;
}

void visSolution(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd& u, const std::string&& filename) {
  const lf::fe::MeshFunctionFE mf_sol(fe_space, u);
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), filename);
  vtk_writer.WritePointData("solution", mf_sol);
}

// Below here are old and inefficient implementations of the IMEXTimestep class
void IMEXTimestep_inefficient::compTimestep(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    double tau, Eigen::VectorXd& y) const {
  // Compute the time dependant non-linear term
  const Eigen::VectorXd r = compNonlinearTerm(fe_test, y);
  // Define Gamma
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;
  const int N = fe_test->LocGlobMap().NumDofs();

  // Define the increment Vectors and store the in the columns of an
  // Eigen::Matrix
  Eigen::MatrixXd k(N, 2);
  Eigen::MatrixXd k_hat(N, 3);

  // Define the butcher Matrices
  Eigen::Matrix2d a;
  a << gamma, 0.0, 1.0 - 2.0 * gamma, gamma;
  const Eigen::Vector2d b{0.5, 0.5};
  Eigen::Matrix3d a_hat;
  a_hat << 0., 0., 0., gamma, 0., 0., gamma - 1., 2.0 * (1.0 - gamma), 0.;
  const Eigen::Vector3d b_hat{0.0, 0.5, 0.5};

  // Define the functions f and g !!!PAY ATTENTION TO THE SIGNS!!!
  auto f = [this, fe_test](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    return -solver_M_.solve(compNonlinearTerm(fe_test, x));
  };
  auto g = [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    return MInvphi_ - MInvA_ * x;
  };

  const Eigen::MatrixXd Id_N = Eigen::MatrixXd::Identity(N, N);

  // Define some temporary helper variables
  Eigen::VectorXd tmp(N);
  tmp.setZero();
  Eigen::VectorXd u(N);

  // We now perform one step of the IMEXRKSSM
  k_hat.col(0) = f(y);
  for (unsigned int i = 0; i < 2; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      tmp += a(i, j) * k.col(j) + a_hat(i + 1, j) * k_hat.col(j);
    }
    tmp += a_hat(i + 1, i) * k_hat.col(i);

    tmp += a(i, i) * MInvphi_;

    u = (Id_N + tau * a(i, i) * MInvA_).lu().solve(y + tau * tmp);
    k.col(i) = g(u);
    k_hat.col(i + 1) = f(u);
    tmp.setZero();
  }
  y += tau * (k * b + k_hat * b_hat);
}

IMEXTimestep_inefficient::IMEXTimestep_inefficient(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test) {
  // Define some helper functions that we can pass to compGalerkinMatrix as
  // alpha, beta and gamma
  const lf::mesh::utils::MeshFunctionConstant<double> mf_one(1.);
  const lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0.);
  auto const_one = [](const Eigen::VectorXd& /*x*/) -> double { return 1.0; };
  auto const_zero = [](const Eigen::VectorXd& /*x*/) -> double { return 0.0; };
  // Flag the edges located on the boundary
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fe_test->Mesh(), 1)};
  auto edges_predicate = [&bd_flags](const lf::mesh::Entity& edge) -> bool {
    return bd_flags(edge);
  };
  // Compute the rhs vector
  const lf::fe::ScalarLoadEdgeVectorProvider<double, decltype(mf_one),
                                             decltype(edges_predicate)>
      LocalVectorAssembler(fe_test, mf_one, edges_predicate);
  // Compute the LHS time independent Matrices
  M_ = compGalerkinMatrix(fe_test->LocGlobMap(), const_zero, const_one,
                          const_zero);
  A_ = compGalerkinMatrix(fe_test->LocGlobMap(), const_one, const_zero,
                          const_one);
  phi_ = lf::assemble::AssembleVectorLocally<Eigen::VectorXd>(
      1, fe_test->LocGlobMap(), LocalVectorAssembler);

  // Do some pre-computations to initialize the sparse solvers
  solver_A_.analyzePattern(A_);
  solver_A_.factorize(A_);
  LF_ASSERT_MSG(solver_A_.info() == Eigen::Success,
                "Solver did not manage to factorize A");

  solver_M_.analyzePattern(M_);
  solver_M_.factorize(M_);
  LF_ASSERT_MSG(solver_M_.info() == Eigen::Success,
                "Solver did not manage to factorize M");

  //  The following solvers are only used in the inefficient time stepping
  //  method Precompute $M^{-1} * A$ and $M^{-1}\phi$
  MInvA_ = solver_M_.solve(A_);
  MInvphi_ = solver_M_.solve(phi_);
}

}  // namespace IMEX
