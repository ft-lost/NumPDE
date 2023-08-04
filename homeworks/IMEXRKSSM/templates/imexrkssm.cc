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
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_test) {
  // ========================================
  // Your code here
  // ========================================
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

  // ========================================
  // Your code here
  // ========================================
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
