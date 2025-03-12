/**
 * @file
 * @brief NPDE homework SimpleLinearFiniteElements
 * @author Amélie Loher
 * @date 11/12/2019
 * @copyright Developed at ETH Zurich
 */

#ifndef SIMPLELINEARFINITEELEMENTS_H_
#define SIMPLELINEARFINITEELEMENTS_H_

#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "tria_mesh_2D.h"

namespace SimpleLinearFiniteElements {

using LocalMatrixHandle_t = std::function<Eigen::Matrix3d(const TriGeo_t &)>;
using FHandle_t = std::function<double(const Eigen::Vector2d &)>;
using LocalVectorHandle_t =
    std::function<Eigen::Vector3d(const TriGeo_t &, FHandle_t)>;

double getArea(const TriGeo_t &triangle);

Eigen::Matrix<double, 2, 3> gradbarycoordinates(const TriGeo_t &vertices);

Eigen::Matrix3d ElementMatrix_Mass_LFE(const TriGeo_t &V);

Eigen::Matrix3d ElementMatrix_Lapl_LFE(const TriGeo_t &V);

Eigen::Matrix3d getElementMatrix(const TriGeo_t &V);

Eigen::Vector3d localLoadLFE(const TriGeo_t &V, const FHandle_t &FHandle);

Eigen::SparseMatrix<double> assembleGalMatLFE(
    const SimpleLinearFiniteElements::TriaMesh2D &Mesh,
    const LocalMatrixHandle_t &getElementMatrix);

Eigen::VectorXd assemLoad_LFE(const TriaMesh2D &Mesh,
                              const LocalVectorHandle_t &getElementVector,
                              const FHandle_t &FHandle);

/**
 * @brief L2Error Computes the L2 error between the approximate solution and
 *                the exact solution
 * @tparam FUNCTOR to be compatible with
 *                 std::function<double(const Eigen::Vector2d &)>
 * @param mesh the mesh to use
 * @param uFEM the solution approximated through FEM
 * @param exact functor object providing the exact solution
 * @return the L2 difference
 */
/* SAM_LISTING_BEGIN_2 */
template <typename FUNCTOR>
double L2Error(const TriaMesh2D &mesh, const Eigen::VectorXd &uFEM,
               FUNCTOR &&exact) {
  double l2error_squared = 0.0;
  double error = 0;
  for(int k = 0; k < mesh._elements.rows(); ++k){
    TriGeo_t triangle = mesh.getVtCoords(k);
    Eigen::Vector3d error_at_v;
    for(int i = 0; i < 3; ++i){
      Eigen::Vector2d coor = mesh._nodecoords.row(mesh._elements(k,i));
      double ex = exact(coor);
      double fem = uFEM[mesh._elements(k,i)];
      error_at_v(i) = ex - fem;
    }
    l2error_squared += getArea(triangle)/3.0 * error_at_v.squaredNorm();
  }
  return std::sqrt(l2error_squared);
}
/* SAM_LISTING_END_2 */

double H1Serror(
    const SimpleLinearFiniteElements::TriaMesh2D &mesh,
    const Eigen::VectorXd &uFEM,
    const std::function<Eigen::Vector2d(const Eigen::Vector2d &)> exact);

std::tuple<Eigen::VectorXd, double, double> Solve(
    const SimpleLinearFiniteElements::TriaMesh2D &mesh);

}  // namespace SimpleLinearFiniteElements

#endif  // SIMPLELINEARFINITEELEMENTS_H_
