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

#if SOLUTION
  // Define the functions g and f
  auto f = [](double x) { return -x * x; };
  auto g = [](double x) { return x; };

  // Define the exact solutions
  auto y_exact = [](double t) { return std::exp(t) / (3 + std::exp(t)); };

  // Keep some temporary variables
  double tmp = 0.0;
  double u;

  // In this for loop we execute the explicit update from $y_0$ to $y_1$
  for (unsigned int step = 1; step < M + 1; ++step) {
    k_hat[0] = f(y_rk[step - 1]);
    // Accumualte the terms all in one variable tmp
    for (unsigned int i = 0; i < 2; ++i) {
      for (unsigned int j = 0; j < i; ++j) {
        tmp += a(i, j) * k[j] + a_hat(i + 1, j) * k_hat[j];
      }
      // Don't forget the additional term
      tmp += a_hat(i + 1, i) * k_hat[i];
      // Compute u
      u = (y_rk[step - 1] + tau * tmp) / (1 - tau * a(i, i));
      // Keep track of the increment vectors
      k[i] = g(u);
      k_hat[i + 1] = f(u);
      // Reset our temporary variable
      tmp = 0;
    }
    // Store the intermediate result and error
    y_rk[step] = y_rk[step - 1] + tau * (b.dot(k) + b_hat.dot(k_hat));
    err[step] = std::abs(y_rk[step] - y_exact(step * tau));
  }
#else
  // ========================================
  // Your code here
  // ========================================
#endif
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

#if SOLUTION
  // Run over all Elements in the mesh
  for (const lf::mesh::Entity* element : mesh_p->Entities(0)) {
    LF_ASSERT_MSG(lf::base::RefEl::kTria() == element->RefEl(),
                  "compNonlinearTerm is only Implemented for triangles");
    // Get the indicies of the corners of the current triangle
    auto indicies = dofh.GlobalDofIndices(*element);
    // Store the coefficients of the basis expansion in $u_{loc}$
    data.u_loc[0] = u[indicies[0]];
    data.u_loc[1] = u[indicies[1]];
    data.u_loc[2] = u[indicies[2]];
    // Compute the local nonlinearTerms
    Eigen::Vector3d local_non_linear_term =
        AssemblerLocalFunc::Eval<data_t, const lf::mesh::Entity>(data,
                                                                 *element);
    nonLinearTerm[indicies[0]] += local_non_linear_term[0];
    nonLinearTerm[indicies[1]] += local_non_linear_term[1];
    nonLinearTerm[indicies[2]] += local_non_linear_term[2];
  }
#else
  // ========================================
  // Your code here
  // ========================================
#endif
  // Return the entire nonLinearTerm
  return nonLinearTerm;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
IMEXTimestep::IMEXTimestep(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test) {
#if SOLUTION
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

  // Do some precomputations to initalize the sparse solvers
  solver_A_.analyzePattern(A_);
  solver_A_.factorize(A_);
  LF_ASSERT_MSG(solver_A_.info() == Eigen::Success,
                "Solver did not manage to factorize A");

  solver_M_.analyzePattern(M_);
  solver_M_.factorize(M_);
  LF_ASSERT_MSG(solver_M_.info() == Eigen::Success,
                "Solver did not manage to factorize M");

  // Precompute $M^-1 * A$
  MInvA_ = solver_M_.solve(A_);
  MInvphi_ = solver_M_.solve(phi_);
#else
  // ========================================
  // Your code here
  // ========================================
#endif
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
void IMEXTimestep::compTimestep(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test,
    double tau, Eigen::VectorXd& y) const {
  // Compute the time dependant non-linear term
  const Eigen::VectorXd r = compNonlinearTerm(fe_test, y);
  // Define Gamma
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;
  const int N = fe_test->LocGlobMap().NumDofs();

  // Define the increment Vectors and store the in the columns of a
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

#if SOLUTION
  // Define the functions f and g !!!PAY ATTENTION TO THE SIGNS!!!
  auto f = [this, r](const Eigen::VectorXd& /*x*/) -> Eigen::VectorXd {
    return -solver_M_.solve(r);
  };
  auto g = [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    return MInvphi_ - MInvA_ * x;
  };

  const Eigen::MatrixXd Id_N = Eigen::MatrixXd::Identity(N, N);

  // Define some temporary helper variables
  Eigen::VectorXd tmp(N);
  tmp.setZero();
  Eigen::VectorXd y_dot(N);
  y_dot = y;
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
  for (unsigned int i = 0; i < 2; ++i) {
    y_dot += tau * (b[i] * k.col(i) + b_hat[i] * k_hat.col(i));
  }
  y_dot += tau * (b_hat[2] * k_hat.col(2));

  y = y_dot;
#else
  // ========================================
  // Your code here
  // ========================================
#endif
}
/* SAM_LISTING_END_4 */

Eigen::VectorXd solveTestProblem(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    int M) {
  IMEXTimestep Timestepper(fe_space);
  const unsigned int N = fe_space->LocGlobMap().NumDofs();
  const double tau = 1. / M;

  Eigen::VectorXd u = Eigen::VectorXd::Constant(N, 0.0);

  for (unsigned int i = 0; i < M; ++i) {
    Timestepper.compTimestep(fe_space, tau, u);
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
}  // namespace IMEX
