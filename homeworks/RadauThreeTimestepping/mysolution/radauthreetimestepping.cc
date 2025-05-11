/**
 * @file radauthreetimestepping.cc
 * @brief NPDE homework RadauThreeTimestepping
 * @author Erick Schulz, edited by Oliver Rietmann
 * @date 08/04/2019
 * @copyright Developed at ETH Zurich
 */

#include "radauthreetimestepping.h"

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

namespace RadauThreeTimestepping {

/**
 * @brief Implementation of the right hand side (time dependent) source vector
 * for the parabolic heat equation
 * @param dofh A reference to the DOFHandler
 * @param time The time at which to evaluate the source vector
 * @returns The source vector at time `time`
 */
/* SAM_LISTING_BEGIN_1 */
Eigen::VectorXd rhsVectorheatSource(const lf::assemble::DofHandler &dofh,
                                    double time) {
  // Dimension of finite element space
  const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
  // Right-hand side vector has to be set to zero initially
  Eigen::VectorXd phi(N_dofs);
  //====================

  auto f = [time] (Eigen::Vector2d x) ->double{
    Eigen::Vector2d v;
    v << x(0) -0.5*std::cos(time*M_PI), x(1) - 0.5 * std::sin(M_PI * time);
    if(v.norm() < 0.5) return 1;
    else return 0;
  };
  auto mesh_p = dofh.Mesh();
  phi.setZero();

  TrapRuleLinFEElemVecProvider<decltype(f)> elvec_builder(f);


  lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, phi);

  auto bd_flags = lf::mesh::utils::flagEntitiesOnBoundary(dofh.Mesh(),2);

  for(const lf::mesh::Entity *v : mesh_p->Entities(2)){
    if(bd_flags(*v)){
      auto dof_idx = dofh.GlobalDofIndices(*v);
      phi(dof_idx[0]) = 0.0;
    }
  }
  //====================
  return phi;
}
/* SAM_LISTING_END_1 */

/**
 * @brief Heat evolution solver: the solver obtains the
 * discrete evolution operator from the Radau3MOLTimestepper class and
 * repeatedly iterates its applicaiton starting from the initial condition
 * @param dofh The DOFHandler object
 * @param m is total number of steps until final time final_time (double)
 * @param final_time The duration for which to solve the PDE
 * @returns The solution at the final timestep
 */
/* SAM_LISTING_BEGIN_6 */
Eigen::VectorXd solveHeatEvolution(const lf::assemble::DofHandler &dofh,
                                   unsigned int m, double final_time) {
  Eigen::VectorXd discrete_heat_sol(dofh.NumDofs());
  //====================
/*  //
  const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
  double tau = final_time/m;
  discrete_heat_sol.setZero();
  double time;
  Radau3MOLTimestepper stepper(dofh);
//  Eigen::VectorXd curr = stepper.discreteEvolutionOperator(0.0, tau, Eigen::VectorXd::Zero(N_dofs));
  Eigen::VectorXd next;
  for(int i = 1; i < m; ++i){
    time = i*tau;
  //  next  = stepper.discreteEvolutionOperator(time, tau, curr);
    curr = next;
  }
  discrete_heat_sol = curr;*/
  //====================
  return discrete_heat_sol;
}
/* SAM_LISTING_END_6 */

/* Implementing member function Eval of class LinFEMassMatrixProvider*/
Eigen::Matrix<double, 3, 3> LinFEMassMatrixProvider::Eval(
    const lf::mesh::Entity &tria) {
  Eigen::Matrix<double, 3, 3> elMat;
  //====================
  // Throw error in case no triangular cell
  LF_VERIFY_MSG(tria.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << tria.RefEl());
  // Compute the area of the triangle cell
  const double area = lf::geometry::Volume(*(tria.Geometry()));
  // Assemble the mass element matrix over the cell
  // clang-format off
  elMat << 2.0, 1.0, 1.0,
           1.0, 2.0, 1.0,
           1.0, 1.0, 2.0;
  // clang-format on
  elMat *= area / 12.0;

  //====================
  return elMat;  // return the local mass element matrix
}

/* Implementing constructor of class Radau3MOLTimestepper */
/* SAM_LISTING_BEGIN_4 */
Radau3MOLTimestepper::Radau3MOLTimestepper(const lf::assemble::DofHandler &dofh)
    : dofh_(dofh) {
  //====================
  // Your code goes here
  // Add any additional members you need in the header file
  const lf::uscalfe::size_type N_dofs(dofh_.NumDofs());

  lf::assemble::COOMatrix<double> A_COO(N_dofs,
                                        N_dofs);  // element matrix Laplace
  lf::assemble::COOMatrix<double> M_COO(N_dofs,
                                        N_dofs);


  auto bd_flags = lf::mesh::utils::flagEntitiesOnBoundary(dofh.Mesh(),2);

  auto selector = [&bd_flags,&dofh ](unsigned int idx) ->bool{
    return bd_flags(dofh.Entity(idx));
  };

  lf::uscalfe::LinearFELaplaceElementMatrix A_elem;
  lf::assemble::AssembleMatrixLocally(0,dofh, dofh, A_elem, A_COO);
  dropMatrixRowsColumns(selector, A_COO);


  LinFEMassMatrixProvider M_elem;
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, M_elem, M_COO);
  dropMatrixRowsColumns(selector, M_COO);

  B << 5.0/12.0, -1.0/12.0, 3.0/4.0, 1.0/4.0;
  c << 1.0/3.0, 1.0;
  b << 0.75, 0.25;

  M_kp = Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(2,2), M);
  A_kp = Eigen::kroneckerProduct(B,A);


  //====================
}
/* SAM_LISTING_END_4 */

/* Implementation of Radau3MOLTimestepper member functions */
// The function discreteEvolutionOperator() returns the discretized evolution
// operator as obtained from the Runge-Kutta Radau IIA 2-stages method using the
// Butcher table as stored in the Radau3MOLTimestepper class
/* SAM_LISTING_BEGIN_5 */
Eigen::VectorXd Radau3MOLTimestepper::discreteEvolutionOperator(
    double time, double tau, const Eigen::VectorXd &mu) const {
  Eigen::VectorXd discrete_evolution_operator(dofh_.NumDofs());
  //====================
  //

  const lf::uscalfe::size_type N_dofs(dofh_.NumDofs());
  Eigen::SparseMatrix<double> LHS = M_kp + tau * A_kp;

  Eigen::VectorXd rhs(2*N_dofs);
  Eigen::VectorXd sub = A * mu;

  rhs << rhsVectorheatSource(dofh_, time + c(0)*tau) - sub, rhsVectorheatSource(dofh_, time + tau) - sub;

  Eigen::SparseLU<Eigen::SparseMatrix<double>>solver_(LHS);
  Eigen::VectorXd k = solver_.solve(rhs);
  discrete_evolution_operator = mu + tau * (b(0) * k.head(N_dofs) + b(1) * k.tail(N_dofs));


  //====================
  return discrete_evolution_operator;
}
/* SAM_LISTING_END_5 */

}  // namespace RadauThreeTimestepping
