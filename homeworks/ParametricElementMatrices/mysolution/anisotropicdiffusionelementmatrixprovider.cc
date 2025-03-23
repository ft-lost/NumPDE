/** @brief NPDE homework ParametricElementMatrices code
 * @author Simon Meierhans, Erick Schulz (refactoring)
 * @date 13/03/2019, 19/11/2019 (refactoring)
 * @copyright Developed at ETH Zurich */

#include "anisotropicdiffusionelementmatrixprovider.h"

namespace ParametricElementMatrices {

/** @brief Compute the local element matrix for the Galerkin matrix of
 *
 *     \int_{\Omega} (1 + d(x)d(x)') * grad(u(x)).grad(v(x)) dx
 *
 * using linear first-order lagrangian finite elements. A local edge-midpoint
 * quadrature rule is used for integration over a cell:
 *
 *   \int_K phi(x) dx = (vol(K)/#vertices(K)) * sum_{edges} phi(midpoint)
 *
 * where K is a cell.
 * @param cell current cell */
Eigen::MatrixXd AnisotropicDiffusionElementMatrixProvider::Eval(
    const lf::mesh::Entity &cell) {
  Eigen::MatrixXd element_matrix;  // local matrix to return

  // Cell data
  auto cell_geometry = cell.Geometry();

  /* SOLUTION_BEGIN */
  // Integration formula distinguishes between triagular and quadrilateral cells
  switch (cell_geometry->RefEl()) {
    /* TRIANGULAR CELL */
    case lf::base::RefEl::kTria(): {
      /* SAM_LISTING_BEGIN_1 */

      // ===================
      Eigen::MatrixXd midpoints = Eigen::MatrixXd(2,3);
      midpoints << 0.5, 0.5, 0, 0, 0.5, 0.5;

      auto k = [=](Eigen::Vector2d x)->Eigen::MatrixXd{
        Eigen::VectorXd d_x = anisotropy_vec_field_(x);
        Eigen::Matrix2d res = Eigen::Matrix2d::Identity(2,2) + d_x * d_x.transpose();
        return res;
      };
      Eigen::MatrixXd Phi_n = cell_geometry->Global(midpoints);
      Eigen::VectorXd detDPhi = cell_geometry->IntegrationElement(midpoints);
      Eigen::MatrixXd DPhi_inv_T = cell_geometry->JacobianInverseGramian(midpoints);

      Eigen::MatrixXd local_grad = Eigen::MatrixXd(2,3);
      local_grad <<  -1.0, 1.0, 0.0, -1.0, 0.0, 1.0;
      element_matrix = Eigen::MatrixXd::Zero(3,3);
      double area = 0.5;

      for(int l = 0; l < 3; ++l){
        Eigen::Matrix2d diff = k(Phi_n.col(l));
        Eigen::MatrixXd gradients_param(2,3);
        gradients_param = DPhi_inv_T.block(0, 2*l, 2, 2) * local_grad;
        for(int i = 0; i < 3; ++i){
          for(int j = 0; j < 3; ++j){
            double integrand = (gradients_param.col(i).transpose() * diff * gradients_param.col(j));

            element_matrix(i,j) += detDPhi(l) * integrand;
          }
        }
      }
      element_matrix *= area/3.0;
      // ===================

      break;
      /* SAM_LISTING_END_1 */
    }

    /* QUADRILATERAL CELL */
    case lf::base::RefEl::kQuad(): {
      /* SAM_LISTING_BEGIN_2 */

      // ===================
      Eigen::MatrixXd midpoints = Eigen::MatrixXd(2,4);
      midpoints << 0.5, 1, 0.5, 0, 0, 0.5, 1.0, 0.5;


      Eigen::MatrixXd Phi_n = cell_geometry->Global(midpoints);
      Eigen::VectorXd detDPhi = cell_geometry->IntegrationElement(midpoints);
      Eigen::MatrixXd DPhi_inv_T = cell_geometry->JacobianInverseGramian(midpoints);


      auto local_grad = [](Eigen::Vector2d x)->Eigen::MatrixXd{
        Eigen::Matrix<double, 2, 4> element_matrix;
        element_matrix(0, 0) = x(1) - 1;
        element_matrix(1, 0) = x(0) - 1;
        element_matrix(0, 1) = 1 - x(1);
        element_matrix(1, 1) = -x(0);
        element_matrix(0, 2) = x(1);
        element_matrix(1, 2) = x(0);
        element_matrix(0, 3) = -x(1);
        element_matrix(1, 3) = 1 - x(0);
        return element_matrix;

      };

      auto k = [=](Eigen::Vector2d x)->Eigen::MatrixXd{
        Eigen::VectorXd d_x = anisotropy_vec_field_(x);
        Eigen::Matrix2d res = Eigen::Matrix2d::Identity(2,2) + d_x * d_x.transpose();
        return res;
      };

      element_matrix = Eigen::MatrixXd::Zero(4,4);
      double area = 1;

      for(int l = 0; l < 4; ++l){
        Eigen::Matrix2d diff = k(Phi_n.col(l));
        Eigen::MatrixXd gradients_param(2,4);
        gradients_param = DPhi_inv_T.block(0, 2*l, 2, 2) * local_grad(midpoints.col(l));
        for(int i = 0; i < 4; ++i){
          for(int j = 0; j < 4; ++j){
            double integrand = (gradients_param.col(i).transpose() * diff * gradients_param.col(j));

            element_matrix(i,j) += detDPhi(l) * integrand;
          }
        }
      }
      element_matrix *= area/4.0;
      // ===================


      break;
      /* SAM_LISTING_END_2 */
    }

    /* ERROR CASE WHERE THE CELL IS NEITHER A TRIANGLE NOR A QUADRILATERAL */
    default:
      LF_VERIFY_MSG(false, "received neither triangle nor quadrilateral");
  }
  /* SOLUTION_END */
  return element_matrix;
}  // AnisotropicDiffusionElementMatrixProvider::Eval

}  // namespace ParametricElementMatrices
