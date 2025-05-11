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
  int N = dofh.NumDofs();
  lf::assemble::COOMatrix<double> M(N, N);

  auto gamma = [](const Eigen::Vector2d& x)->double{
    return 1;
  };
  auto alpha = [](const Eigen::Vector2d& x)->double{
    return 0;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_gamma{gamma};
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};

  lf::uscalfe::ReactionDiffusionElementMatrixProvider<double,decltype(mf_alpha),decltype(mf_gamma)> M_elem(fe_space, mf_alpha, mf_gamma);
  lf::assemble::AssembleMatrixLocally<lf::assemble::COOMatrix<double>>(0, dofh, dofh, M_elem, M);

  auto bd_flag = lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,2);

  auto set = [&bd_flag, &dofh](int i, int j){
    return bd_flag(dofh.Entity(i));
  };
  M.setZero(set);




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
  int N = dofh.NumDofs();
  lf::assemble::COOMatrix<double> A(N, N);

  auto gamma = [](const Eigen::Vector2d& x)->double{
    return 0;
  };
  auto alpha = [](const Eigen::Vector2d& x)->double{
    return 1;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_gamma{gamma};
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};

  lf::uscalfe::ReactionDiffusionElementMatrixProvider<double,decltype(mf_alpha),decltype(mf_gamma)> A_elem(fe_space, mf_alpha, mf_gamma);
  lf::assemble::AssembleMatrixLocally<lf::assemble::COOMatrix<double>>(0, dofh, dofh, A_elem, A);

  auto bd_flag = lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,2);

  auto set = [&bd_flag, &dofh](int i, int j){
    return bd_flag(dofh.Entity(i));
  };
  A.setZero(set);
  for (int i = 0; i < dofh.NumDofs(); ++i) {
    if (bd_flag(dofh.Entity(i))) A.AddToEntry(i, i, 1.0);
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
  auto N = dofh.NumDofs();
  auto bd_flag = lf::mesh::utils::flagEntitiesOnBoundary(dofh.Mesh(),2);
  zero_one = Eigen::VectorXd::Zero(N);
  for(int i = 0; i < N; ++i){
    if(bd_flag(dofh.Entity(i))) zero_one(i) += 1;
  }


  //====================
}

Eigen::VectorXd RHSProvider::operator()(double t) const {
  //====================
  // Your code goes here
  // Replace this dummy return value:
  return  g_(t)*zero_one;

  //====================
}
/* SAM_LISTING_END_3 */

}  // namespace GaussLobattoParabolic
