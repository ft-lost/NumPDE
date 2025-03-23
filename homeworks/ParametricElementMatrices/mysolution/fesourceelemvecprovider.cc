/** @brief NPDE homework ParametricElementMatrices code
 * @author Simon Meierhans, Erick Schulz (refactoring)
 * @date 13/03/2019, 19/11/2019 (refactoring)
 * @copyright Developed at ETH Zurich */

#include "fesourceelemvecprovider.h"

namespace ParametricElementMatrices {

/** @brief Compute the element vector for
 *
 *       \int_{\Omega} v(x)/(1 + w(x)^2) dx
 *
 * using linear first-order lagrangian finite elements. A local edge-midpoint
 * quadrature rule is used for integration over a cell:
 *
 *   \int_K phi(x) dx = (vol(K)/#vertices(K)) * sum_{edges} phi(midpoint)
 *
 * where K is a cell.
 * @param cell current cell */
Eigen::VectorXd FESourceElemVecProvider::Eval(const lf::mesh::Entity &cell) {
  Eigen::VectorXd element_vector;  // local vector to return;

  /* TOOLS AND DATA */
  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
  // Obtain cell data
  auto cell_geometry = cell.Geometry();
  auto cell_global_idx = dofh.GlobalDofIndices(cell);

  /* SOLUTION_BEGIN */
  // Integration formula distinguishes between triagular and quadrilateral cells
  switch (cell_geometry->RefEl()) {
    case lf::base::RefEl::kTria(): {
      /* SAM_LISTING_BEGIN_1 */

      // ===================
      double area = 0.5;
      Eigen::MatrixXd midpoints = Eigen::MatrixXd(2,3);
      midpoints << 0.5, 0.5, 0, 0, 0.5, 0.5;
      element_vector = Eigen::VectorXd::Zero(3);
      Eigen::VectorXd w = Eigen::VectorXd::Zero(3);

      Eigen::MatrixXd Phi_n = cell_geometry->Global(midpoints);
      Eigen::VectorXd detDPhi = cell_geometry->IntegrationElement(midpoints);

      for(int i = 0; i < 3; ++i){
        w(i) = coeff_expansion_(cell_global_idx[i]);
      }


      for(int i = 0;  i < 3; ++i){
        element_vector(i) +=
            detDPhi(i) *
            (0.5 / (1. + std::pow(0.5 * w(i) + 0.5 * w((i + 1) % 3), 2)));
        element_vector((i + 1) % 3) +=
            detDPhi(i) *
            (0.5 / (1. + std::pow(0.5 * w(i) + 0.5 * w((i + 1) % 3), 2)));

      }
      element_vector *= area/3.0;

      // ===================

      break;
      /* SAM_LISTING_END_1 */
    }
    case lf::base::RefEl::kQuad(): {
      /* SAM_LISTING_BEGIN_2 */

      // ===================
      double area = 1.0;
      Eigen::MatrixXd midpoints = Eigen::MatrixXd(2,4);
      midpoints << 0.5, 1, 0.5, 0, 0, 0.5, 1.0, 0.5;
      element_vector = Eigen::VectorXd::Zero(4);
      Eigen::VectorXd w = Eigen::VectorXd::Zero(4);

      Eigen::MatrixXd Phi_n = cell_geometry->Global(midpoints);
      Eigen::VectorXd detDPhi = cell_geometry->IntegrationElement(midpoints);

      for(int i = 0; i < 4; ++i){
        w(i) = coeff_expansion_(cell_global_idx[i]);
      }


      for(int i = 0; i < 4; ++i){
        element_vector(i) +=
            detDPhi(i) *
            (0.5 / (1. + std::pow(0.5 * w(i) + 0.5 * w((i + 1) % 4), 2)));
        element_vector((i + 1) % 4) +=
            detDPhi(i) *
            (0.5 / (1. + std::pow(0.5 * w(i) + 0.5 * w((i + 1) % 4), 2)));

      }
      element_vector *= area/4.0;
      break;
      // ===================

      /* SAM_LISTING_END_2 */
    }
      /* ERROR CASE WHERE THE CELL IS NEITHER A TRIANGLE NOR A QUADRILATERAL */
    default:
      LF_VERIFY_MSG(false, "received neither triangle nor quadrilateral");
      /* SOLUTION_END */
  }
  return element_vector;
}
}  // namespace ParametricElementMatrices
