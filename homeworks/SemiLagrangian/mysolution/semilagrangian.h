#ifndef SEMILAGRANGIAN_H
#define SEMILAGRANGIAN_H
/**
 * @file semilagrangian.h
 * @brief NPDE homework TEMPLATE HEADER FILE
 * @author
 * @date May 2022
 * @copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Core>
#include <cmath>
#include <iostream>

namespace SemiLagrangian {

/**
 * @brief Evaluates the solution of the transport problem using the solution
 * formula and Heun's method
 * @param x location
 * @param K maximum number of steps of Heun's method
 * @param t time
 * @param v velocity field (time independent)
 * @param u0 initial condition
 * @return $u(x,t)$
 */
/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTOR_V, typename FUNCTOR_U0>
double solveTransport(const Eigen::Vector2d& x, int K, double t, FUNCTOR_V&& v,
                      FUNCTOR_U0&& u0) {
  //====================
  // Your code goes here
  //====================
  return 0.0;
}
/* SAM_LISTING_END_1 */
/**
 * @brief Evaluates an FE function u_h
 * @param x point coordinates
 * @param u FE basis coefficients of u_h
 * @return u_h(x)
 */
double evalFEfunction(const Eigen::Vector2d& x, const Eigen::VectorXd& u);

/**
 * @brief Computes the interior nodes of the mesh
 * @param M number of cells in one direction
 * @return 2x(M-1)^2 matrix containing the interior nodes of the mesh as columns
 */
Eigen::MatrixXd findGrid(int M);

/**
 * @brief Evaluates the source term of the semi-lagrangian scheme.
 * @param u_old FE basis coefficients of the solution at the previous timestep
 * @param tau step size
 * @param velocity velociy field
 * @return RHS-vector of the scheme
 */
/* SAM_LISTING_BEGIN_2 */
template <typename FUNCTOR>
Eigen::VectorXd semiLagrangeSource(const Eigen::VectorXd& u_old, double tau,
                                   FUNCTOR&& velocity) {
  // Note: components of coefficient vectors are associated
  // with interior nodes only
  int N = u_old.size();
  // Extract number of cells in one direction
  int root = std::round(std::sqrt(N));
  if (N != root * root) {
    std::cerr << "The number of dofs should be a perfect square!" << std::endl;
  }
  int M = root + 1;
  Eigen::MatrixXd grid = findGrid(M);
  Eigen::VectorXd f(N);

  //====================
  // Your code goes here
  //====================
  return f;
}
/* SAM_LISTING_END_2 */

/**
 * @brief Solves the pure transport problem using the semi-lagrangian scheme.
 * @param M number of cells in one direction
 * @param K total number of equidistant timesteps
 * @param T final time
 * @param rec takes in an Eigen::VectorXd as argument and stores for later
 * visualizations
 * @return FE Basis coefficients of the approximate solution at final time
 */

/* SAM_LISTING_BEGIN_3 */
template <typename RECORDER = std::function<void(const Eigen::VectorXd&)>>
Eigen::VectorXd semiLagrangePureTransport(
    int M, int K, double T,
    RECORDER&& rec = [](const Eigen::VectorXd&) -> void {}) {
  int N = (M - 1) * (M - 1);  // internal dofs

  // Coordinates of nodes of the grid
  Eigen::MatrixXd grid = findGrid(M);
  Eigen::VectorXd u(N);

  //====================
  // Your code goes here
  //====================
  return u;
}
/* SAM_LISTING_END_3 */

void testFloorAndDivision();

/**
 * @brief Visualizes the pure transport problem.
 * @param M number of cells in one direction
 * @param K total number of equidistant timesteps
 * @param T final time
 */
void SemiLagrangeVis(int M, int K, double T);

}  // namespace SemiLagrangian

#endif  // SEMILAGRANGIAN_H
