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
  // The formulas for the columns of the jacobian have been derived in
  // \prbautoref{sp:5}
  J.col(0) =
      gamma01.derivative(point[0]) + gamma12(point[1]) -
      point[1] * gamma12.derivative(1. - point[0]) -
      (gamma01(1. - point[1]) + gamma20(1. - point[1]) - gamma01(0.)) -
      point[1] * (gamma01.derivative(point[0]) + gamma20.derivative(point[0]));
  J.col(1) = -gamma20.derivative(1. - point[1]) +
             point[0] * gamma12.derivative(point[1]) + gamma12(1. - point[0]) +
             point[0] * (gamma01.derivative(1. - point[1]) +
                         gamma20.derivative(1. - point[1])) -
             (gamma01(point[0]) + gamma20(point[0]) - gamma01(0.));
  return J;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
Eigen::MatrixXd evalBlendLocMat(const Curve& gamma01, const Curve& gamma12,
                                const Curve& gamma20) {
  // Variable for returning $3\times 3$ element matrix
  Eigen::MatrixXd lclMat(3, 3);
  lclMat.setZero();
  Eigen::MatrixXd xi(2, 3);  // coordinates of midpoints of curves
  xi.col(0) << 0.5, 0.;
  xi.col(1) << 0.5, 0.5;
  xi.col(2) << 0., 0.5;

  // (constant) gradients of barycentric coordinate function on the reference
  // element, which are the preimages of the local shape functions under
  // pullback
  Eigen::MatrixXd gradEval(3, 2);
  gradEval << -1., -1.,  // $\grad \wh{\lambda}_1$
      1., 0.,            // $\grad \wh{\lambda}_2$
      0., 1.;            // $\grad \wh{\lambda}_3$
  // Generate element matrix by adding up rank-1 matrices formed from gradients
  // of local shape functions at midpoints of edges.
  for (int l = 0; l < xi.cols(); ++l) {
    Eigen::Vector2d xi_l = xi.col(l);
    // Call auxiliary function implemented in \prbautoref{sp:6}
    Eigen::MatrixXd Ji_l = JacobianPhi(xi_l, gamma01, gamma12, gamma20);
    double detJi_l = std::abs(Ji_l.determinant());
    // Transformation matrix for gradients, see \lref{lem:Gtrf}
    Eigen::MatrixXd invJT_l = Ji_l.inverse().transpose();
    // Transformed gradient
    // Eigen::MatrixXd grad_b_l = gradEval * invJT_l;
    // Rank-1 update of element matrix
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        lclMat(i, j) += detJi_l *
                        (invJT_l * gradEval.row(i).transpose()).transpose() *
                        (invJT_l * gradEval.row(j).transpose());
      }
    }
  }
  // Don't forget the quadrature weight $\frac{1}{6}$: area
  // of the reference triangle $=\frac12$!
  return lclMat / 6.;
}
/* SAM_LISTING_END_2 */

}  // namespace BlendedParameterization
