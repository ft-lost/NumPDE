/**
 * @file imexrkssm_test.cc
 * @brief NPDE homework IMEXRKSSM code
 * @author Bob Schreiner
 * @date July 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../imexrkssm.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace IMEX::test {
TEST(IMEX, IMEXError) {
  Eigen::VectorXd M(8);
  M[0] = 8;
  Eigen::VectorXd rk_err(8);
  for (unsigned int i = 0; i < 7; ++i) {
    rk_err[i] = IMEX::IMEXError(M[i]);
    M[i + 1] = 2 * M[i];
  }
  rk_err[7] = IMEX::IMEXError(M[7]);
  Eigen::VectorXd reference_error(8);
  reference_error << 8.38411e-05, 9.30965e-06, 1.10098e-06, 1.3398e-07,
      1.65278e-08, 2.05249e-09, 2.55726e-10, 3.19141e-11;
  ASSERT_NEAR(0.0, (rk_err - reference_error).norm(), 1e-6);
}

TEST(IMEX, compNonlinearTerm) {
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/square.msh");
  auto mesh = reader.mesh();
  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
  const int N = fe_space->LocGlobMap().NumDofs();
  Eigen::VectorXd r_reference(N);

  r_reference << 0.000405321, 0.000405321, 0.000405321, 0.000405321,
      0.000480734, 0.000493396, 0.000837522, 0.000546695, 0.000928083,
      0.000500823, 0.00048208, 0.000481043, 0.000495676, 0.000847989,
      0.000516086, 0.000943214, 0.000502351, 0.000482988, 0.000480734,
      0.000493396, 0.000835133, 0.000520099, 0.000984481, 0.000511401,
      0.000483659, 0.000481064, 0.00049593, 0.000860141, 0.00052019, 0.00090337,
      0.000500667, 0.000482058, 0.00111799, 0.00148793, 0.00108588, 0.00116023,
      0.00113989, 0.00159491, 0.00169029, 0.00171437, 0.00135651, 0.00113733,
      0.00102207, 0.00106487, 0.000960176, 0.00157273, 0.00146917, 0.00142495,
      0.0014959, 0.00119381, 0.0012383, 0.00166715, 0.00173303, 0.00127295,
      0.0011935, 0.0014753, 0.00144151, 0.00151922, 0.0012044, 0.00146358,
      0.00140397, 0.00100777, 0.00102279, 0.000970461, 0.00104236, 0.00117915,
      0.000950332, 0.00106768, 0.00110405, 0.00110275, 0.00106767, 0.00112349,
      0.00111326, 0.00107575, 0.00111315, 0.000921448, 0.000931092, 0.0011238,
      0.000808141, 0.000939637, 0.000936111, 0.000930794, 0.000930766,
      0.000829705, 0.000916933, 0.000916826, 0.000916691, 0.000918927,
      0.0010087, 0.00079212, 0.000742585, 0.000990985, 0.000852664, 0.000939443,
      0.00105427, 0.000744762, 0.000806803, 0.000808132, 0.000954945,
      0.000683306, 0.000684083, 0.000683996, 0.000683306, 0.000691912,
      0.000695767, 0.000686573, 0.000686641;

  const Eigen::VectorXd y = Eigen::VectorXd::Constant(N, 0.1);
  const Eigen::VectorXd r = compNonlinearTerm(fe_space, y);
  ASSERT_NEAR(0.0, (r - r_reference).norm(), 1e-6);
}

TEST(IMEX, TimeStepTest) {
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/square.msh");
  auto mesh = reader.mesh();
  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
  const int N = fe_space->LocGlobMap().NumDofs();
  Eigen::VectorXd u_ref = Eigen::VectorXd::Zero(N);

  int M = std::pow(2, 8);

  Eigen::VectorXd u = Eigen::VectorXd::Zero(N);

  auto initial_values = [](const Eigen::Vector2d& x) {
    return std::sin(x(0) * 2.0 * M_PI) * std::sin(x(1) * 2.0 * M_PI) + 1.0;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_init(initial_values);
  lf::fe::ScalarLoadElementVectorProvider element_vector_provider(fe_space,
                                                                  mf_init);
  lf::assemble::AssembleVectorLocally(0, fe_space->LocGlobMap(),
                                      element_vector_provider, u);
  lf::assemble::AssembleVectorLocally(0, fe_space->LocGlobMap(),
                                      element_vector_provider, u_ref);

  const double tau = 1. / M;
  const double gamma = (3.0 + std::sqrt(3)) / 6.0;
  // Define the butcher Matrix a
  Eigen::Matrix2d a;
  a << gamma, 0.0, 1.0 - 2.0 * gamma, gamma;

  // Create the timestepper
  IMEXTimestep Timestepper(fe_space, tau, a.diagonal());
  IMEXTimestep_inefficient InefficientTimestepper(fe_space);

  for (unsigned int i = 0; i < M; ++i) {
    Timestepper.compTimestep(fe_space, u);
    InefficientTimestepper.compTimestep(fe_space, tau, u_ref);
  }

  ASSERT_NEAR(0.0, (u - u_ref).array().abs().maxCoeff(), 1e-7);
}

}  // namespace IMEX::test
