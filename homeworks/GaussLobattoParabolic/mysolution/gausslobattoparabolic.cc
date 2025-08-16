/**
 * @file gausslobattoparabolic.cc
 * @brief NPDE exam TEMPLATE CODE FILE
 * @author Oliver Rietmann
 * @date 22.07.2020
 * @copyright Developed at SAM, ETH Zurich
 */

#include "gausslobattoparabolic.h"

#include <lf/assemble/assemble.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseLU>
#include <functional>
#include <memory>
#include <utility>




namespace GaussLobattoParabolic {

/* SAM_LISTING_BEGIN_1 */
lf::assemble::COOMatrix<double> initMbig(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space) {
  const lf::assemble::DofHandler &dofh = fe_space->LocGlobMap();
  //====================
  // Your code goes here
  // Replace this dummy assignment for M:
  //
  auto mesh_p = dofh.Mesh();
  unsigned int N = dofh.NumDofs();
  Eigen::SparseMatrix<double> M(N,N);
  lf::mesh::utils::MeshFunctionConstant<double> mf_one(1.);

  lf::fe::MassElemenetMatrixProvider<double, decltype<mf_one>> Elem_M(fe_space,mf_one);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, Elem_M, M);

  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags =
    lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,2);

  auto selector = [&bd_flags, &dofh](int i, int j)->bool{
    return bd_flags(dofh.Entity(i));
  };
  M.setZero(selector);


  //====================

  return M;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
lf::assemble::COOMatrix<double> initAbig(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space){
  const lf::assemble::DofHandler &dofh = fe_space->LocGlobMap();
  //====================
  // Your code goes here
  // Replace this dummy assignment for A:
  auto mesh_p = dofh.Mesh();
  unsigned int N = dofh.NumDofs();
  Eigen::SparseMatrix<double> A(N,N);
  lf::mesh::utils::MeshFunctionConstant<double> mf_one(1.);
  lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0.);

  lf::fe::ReactionDiffusionElemenetMatrixProvider<double, decltype<mf_one>, decltype<mf_zero>> Elem_A(fe_space,mf_one, mf_zero);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, Elem_M, A);

  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags =
    lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,2);

  auto selector = [&bd_flags, &dofh](int i, int j)->bool{
    return bd_flags(dofh.Entity(i));
  };
  A.setZero(selector);
  for(int i = 0; i < N; ++i){
    if(bd_flags(dofh.Entity(i))){
      A.AddToEntry(i,i,1.0);
    }
  }

  //====================

  return A;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
RHSProvider::RHSProvider(const lf::assemble::DofHandler &dofh,
                         std::function<double(double)> g)
    : g_(std::move(g)) {
  //====================
  int N = dofh.NumDofs();
   lf::mesh::utils::CodimMeshDataSet<bool> bd_flags =
    lf::mesh::utils::flagEntitiesOnBoundary(dofh.mesh,2);
   zero_one = Eigen::VectorXd::Zero(N);
   for(int i = 0; i < N; ++i){
     if(bd_flag(dofh.Entity(i))){
       zero_one(i) = 1;
     }
   }
   g_ = g;

  //====================
}

Eigen::VectorXd RHSProvider::operator()(double t) const {
  //====================
  // Your code goes here
  // Replace this dummy return value:
  return g_(t) * zero_one;

  //====================
}
/* SAM_LISTING_END_3 */

}  // namespace GaussLobattoParabolic
