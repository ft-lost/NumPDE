/*
 * @file
 * @brief NPDE homework NonConformingCrouzeixRaviartFiniteElements code
 * @author Anian Ruoss, edited Amélie Loher
 * @date   18.03.2019, 03.03.20
 * @copyright Developed at ETH Zurich
 */

#ifndef NUMPDE_SOLVE_CR_NEUMANN_BVP_H
#define NUMPDE_SOLVE_CR_NEUMANN_BVP_H

#include <lf/assemble/assemble.h>
#include <lf/uscalfe/uscalfe.h>

#include "crfespace.h"

namespace NonConformingCrouzeixRaviartFiniteElements {

/* SAM_LISTING_BEGIN_1 */
template <typename GAMMA_COEFF, typename F_FUNCTOR>
Eigen::VectorXd solveCRNeumannBVP(std::shared_ptr<CRFeSpace> fe_space,
                                  GAMMA_COEFF &&gamma, F_FUNCTOR &&f) {
  Eigen::VectorXd sol;
// TODO: task 2-14.u)
  //====================
  const lf::asemble::DofHandler &dof_handler{fe_space->LocGlobMap()};
  const size_type num_dofs = dof_handler.NumDofs();

  lf::mesh::utils::MeshFunctionGlobal mf_one{
    [](Eigen::Vector2d x) -> double{return 1.;}
  };
  lf::mesh::utils::MeshFunctionGlobal mf_gamma{gamma};
  lf::mesh::utils::MeshFunctionGlobal mf_f{f};

  lf::assemble::COOMatrix<double> A(num_dofs, num_dofs);

  lf::uscale::ReactionDiffusionElementMatrixProvider<double, decltype(mf_one), decltype(mf_gamma)>
    element_matrix_builder(fe_space, mf_gammma);

  lf::assemble::AssembleMatrixLocally(0, dof_handler, dof_handler, element_matrix_builder, A);

  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(num_dofs);
  phi.setZero();

  lf::uscale::ScalarLoadElementVectorProvider<double, decltype(mf_f)>
    load_vector_builder(fe_space, mf_f);

  lf::assemble::AssembleVectorLocally(0, dof_handler, load_vector_builder, phi);

  Eigen::SparseMatrix<double> A_crs = A.makeSparse();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  sol = solver.solve(phi);
  //====================
  return sol;
}
/* SAM_LISTING_END_1 */

}  // namespace NonConformingCrouzeixRaviartFiniteElements

#endif  // NUMPDE_SOLVE_CR_NEUMANN_BVP_H
