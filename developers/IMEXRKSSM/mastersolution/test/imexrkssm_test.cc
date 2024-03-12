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
  Eigen::VectorXd u_ref(N);
  int M = 1024;
  u_ref << 0.0434425, 0.0434457, 0.0434434, 0.0434492, 0.0293169, 0.0309144,
      0.0210828, 0.0298132, 0.0198744, 0.0307305, 0.0292886, 0.0293121,
      0.0308709, 0.0207334, 0.0309517, 0.0192963, 0.0307029, 0.0292573,
      0.0293152, 0.0309175, 0.0210519, 0.0308338, 0.018709, 0.0304284,
      0.0292672, 0.0293112, 0.0308736, 0.0205944, 0.0307722, 0.0201742,
      0.0306995, 0.0292917, 1.86984e-05, 0.000399992, -2.91491e-05, 0.000185663,
      0.000241825, 0.000971097, 0.000896618, 0.000938677, 0.00102834,
      0.00135137, 0.00146696, 0.00142012, 0.00151964, -0.000131932, 8.86764e-06,
      -0.000159586, -0.00210476, -0.00241515, -0.00214354, -0.00200367,
      -0.00199297, -0.00217797, -0.0024125, -0.0021389, -6.59751e-05,
      -0.00012574, -7.67701e-05, 0.00013639, 0.000179789, -0.00462553,
      -0.00367534, -0.00410778, -0.00369735, 0.000162301, 9.24702e-05,
      -0.00454552, -0.00442071, -0.00442195, -0.00453451, -0.00447883,
      -0.00443405, -0.0045019, -0.00432522, 4.99375e-05, 2.77087e-05,
      7.22009e-05, 4.92161e-06, -0.00337608, -0.00335778, -0.00341243,
      -0.00350629, 5.42904e-05, -0.00957099, -0.00957767, -0.00957913,
      -0.00955876, 0.000341966, 0.000354395, 0.0003674, 0.000361194,
      -4.52379e-05, 0.000231246, 0.000312819, 0.000306226, 0.000312481,
      -5.29099e-06, 2.43659e-05, -0.00315849, -0.00317134, -0.0031774,
      -0.00316244, -0.00311926, -0.00302435, -0.0030816, -0.00308685;

  const IMEXTimestep Timestepper(fe_space);
  const double tau = 1. / M;

  Eigen::VectorXd u = Eigen::VectorXd::Constant(N, 0.0);
  Timestepper.compTimestep(fe_space, tau, u);

  ASSERT_NEAR(0.0, (u - u_ref).norm(), 1e-6);
}

}  // namespace IMEX::test
