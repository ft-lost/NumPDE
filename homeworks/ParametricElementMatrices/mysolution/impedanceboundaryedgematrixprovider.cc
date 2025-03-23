/** @brief NPDE homework ParametricElementMatrices code
 * @author Simon Meierhans, Erick Schulz (refactoring)
 * @date 13/03/2019, 19/11/2019 (refactoring)
 * @copyright Developed at ETH Zurich */

#include "impedanceboundaryedgematrixprovider.h"

namespace ParametricElementMatrices {

/* SAM_LISTING_BEGIN_0 */
ImpedanceBoundaryEdgeMatrixProvider::ImpedanceBoundaryEdgeMatrixProvider(
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space,
    Eigen::VectorXd coeff_expansion)
    : fe_space_(fe_space), coeff_expansion_(coeff_expansion) {
  auto mesh = fe_space->Mesh();
  // Obtain an array of boolean flags for the edges of the mesh: 'true'
  // indicates that the edge lies on the boundary. This predicate will
  // guarantee that the computations are carried only on the boundary edges
  bd_flags_ = std::make_shared<lf::mesh::utils::CodimMeshDataSet<bool>>(
      lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1));
}
/* SAM_LISTING_END_0 */

/* SAM_LISTING_BEGIN_1 */
bool ImpedanceBoundaryEdgeMatrixProvider::isActive(
    const lf::mesh::Entity &edge) {
  bool is_bd_edge;
  //====================
  is_bd_edge = (*bd_flags_)(edge);
  //====================
  return is_bd_edge;
}
/* SAM_LISTING_END_1 */

/**  @brief Compute the local edge element matrix for the Galerkin matrix of
 *
 *           \int_{\boundary \Omega} w(x)^2 u(x) v(x) dx
 *
 * @param edge current edge */
/* SAM_LISTING_BEGIN_2 */
Eigen::MatrixXd ImpedanceBoundaryEdgeMatrixProvider::Eval(
    const lf::mesh::Entity &edge) {
  Eigen::MatrixXd element_matrix(2, 2);

  //====================
  element_matrix = Eigen::MatrixXd::Zero(2,2);
  //if(!isActive(edge)) return element_matrix;
  auto geo_p = edge.Geometry();
  double length = lf::geometry::Volume(*geo_p);
  Eigen::VectorXd w(2);
  const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
  auto cell_gi = dofh.GlobalDofIndices(edge);
  w << coeff_expansion_(cell_gi[0]), coeff_expansion_(cell_gi[1]);
  Eigen::MatrixXd A1(2,2);
  A1 << 24, 6, 6, 4;
  Eigen::MatrixXd A2(2,2);
  A2 << 6, 4, 4, 6;
  Eigen::MatrixXd A3(2,2);
  A3 << 4, 6, 6, 24;

  element_matrix = w(0)*w(0)*A1 + 2*w(1)*w(0)* A2 + w(1)*w(1)*A3;
  element_matrix *= length/120.0;

  //====================

  return element_matrix;
}
/* SAM_LISTING_END_2 */

}  // namespace ParametricElementMatrices
