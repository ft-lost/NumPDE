/**
 * @file
 * @brief NPDE homework NonConformingCrouzeixRaviartFiniteElements code
 * @author Anian Ruoss, edited Amélie Loher
 * @date   18.03.2019, 03.03.20
 * @copyright Developed at ETH Zurich
 */

#ifndef NUMPDE_COMPUTE_CR_L2_ERROR_H
#define NUMPDE_COMPUTE_CR_L2_ERROR_H

#include <lf/assemble/assemble.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>

#include "crfespace.h"

namespace NonConformingCrouzeixRaviartFiniteElements {

/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTION>
double computeCRL2Error(std::shared_ptr<CRFeSpace> fe_space,
                        const Eigen::VectorXd &mu, FUNCTION &&u) {
  double l2_error = 0.;

// TODO: task 2-14.w)
  //====================
  //
  
  const lf::assemble::DofHandler &dof_handler{fe_space->LocGlobMap()};
  auto mesh_ptr = fe_space->Mesh();

  for(const lf::mesh::Entity *cell : mesh_ptr->Entities(0)){
    lf::assemble::size_type num_nodes = cell->RefEl().NumNodes();
    LF_ASSERT_MSG(num_nodes == 3, "Only for triangles!");

    const lf::geometry::Geometry *geo_ptr = cell->Geometry();

    const Eigen::MatrixXd vertices = lf::geometry::Corners(*geo_ptr);

    Eigen::MatrixXd midpoints = vertices *(Eigen::Matrix<double, 3, 3>(3,3) <<
                                           0.5, 0.0, 0.5,
                                           0.5, 0.5, 0.0,
                                           0.0, 0.5, 0.5).finished();

    auto cell_idx = dof_handler.GlobalDofIndices(*cell);

    double local_sum = 0;

    for(int i = 0; i < num_nodes; ++i){
      local_sum += std::pow(mu[cell_idx[i]] - u(midpoints.col(i)), 2);
    }
    l2_error += lf::geometry::Volume(*geo_ptr) * (local_sum/num_nodes);
     
  }
  //====================
  return std::sqrt(l2_error);
}
/* SAM_LISTING_END_1 */

}  // namespace NonConformingCrouzeixRaviartFiniteElements

#endif  // NUMPDE_COMPUTE_CR_L2_ERROR_H
