/**
 * @file irkdegenerateevl.cc
 * @brief NPDE homework IRKDegenerateEvl code
 * @author R. Hiptmair
 * @date June 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#include "irkdegenerateevl.h"

#include <lf/assemble/assembler.h>
#include <lf/assemble/coomatrix.h>
#include <lf/assemble/dofhandler.h>
#include <lf/fe/fe_tools.h>
#include <lf/fe/loc_comp_ellbvp.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/utils/codim_mesh_data_set.h>
#include <lf/mesh/utils/mesh_function_global.h>
#include <lf/mesh/utils/special_entity_sets.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace IRKDegenerateEvl {
/* SAM_LISTING_BEGIN_1 */
lf ::assemble::COOMatrix<double> buildM(
    std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<double>> fes_p) {
  // Extract mesh
  const std::shared_ptr<const lf::mesh::Mesh> mesh_p{fes_p->Mesh()};
  // Fetch DofHandler
  const lf::assemble::DofHandler &dofh{fes_p->LocGlobMap()};
  const size_t N = dofh.NumDofs();
  // Sparse matrix in triplet format
  lf::assemble::COOMatrix<double> M_coo(N, N);
// **************************************************
   auto coeff = [](const Eigen::Vector2d)->double{
    return 1;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_coeff{coeff};


  auto bd_flags = lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,1);

  lf::uscalfe::MassEdgeMatirxProvider<double, decltype(mf_coeff), decltype(bd_flags)> M_elem(fes_p, mf_coeff, bd_flags);
  lf::assemble::AssembleMatrixLocally(1,dofh, dofh, M_elem, M_coo);



 ////  ************************************************** */
  return M_coo;
}
/* SAM_LISTING_END_1 */

lf::assemble::COOMatrix<double> buildA(
    std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<double>> fes_p) {
  // Extract mesh
  const std::shared_ptr<const lf::mesh::Mesh> mesh_p{fes_p->Mesh()};
  const lf::mesh::Mesh &mesh{*(fes_p->Mesh())};
  // Fetch DofHandler
  const lf::assemble::DofHandler &dofh{fes_p->LocGlobMap()};
  const size_t N = dofh.NumDofs();
  // Set up ENTITY_MATRIX_PROVIDER
  lf::mesh::utils::MeshFunctionGlobal mf_one{
      [](Eigen::Vector2d /*x*/) -> double { return 1.0; }};
  lf::fe::DiffusionElementMatrixProvider<double, decltype(mf_one)> demp(fes_p,
                                                                        mf_one);
  // Assembly of matrix A
  lf::assemble::COOMatrix<double> A_coo(N, N);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, demp, A_coo);
  return A_coo;
}

}  // namespace IRKDegenerateEvl
