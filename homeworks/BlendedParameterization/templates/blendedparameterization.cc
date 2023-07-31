/**
 * @file BlendedParameterization.cc
 * @brief NPDE homework BlendedParameterization code
 * @author R. Hiptmair
 * @date January 2018
 * @copyright Developed at SAM, ETH Zurich
 */

#include "blendedparameterization.h"

namespace BlendedParameterization {

// Jacobian of blended parameterization mapping
/* SAM_LISTING_BEGIN_1 */
Eigen::MatrixXd JacobianPhi(const coord_t& point, const Curve& gamma01,
                            const Curve& gamma12, const Curve& gamma20) {
  Eigen::MatrixXd J(2, 2);  // Variable for returning Jacobian
  //====================
  // Your code goes here
  //====================
  return J;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
Eigen::MatrixXd evalBlendLocMat(const Curve& gamma01, const Curve& gamma12,
                                const Curve& gamma20) {
  // Variable for returning $3\times 3$ element matrix
  Eigen::MatrixXd lclMat(3, 3);
  lclMat.setZero();

  //====================
  // Your code goes here
  //====================
  return lclMat;
}
/* SAM_LISTING_END_2 */

}  // namespace BlendedParameterization
