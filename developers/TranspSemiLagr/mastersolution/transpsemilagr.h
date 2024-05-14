/**
 * @file transpsemilagr.h
 * @brief NPDE homework TranspSemiLagr code
 * @author Philippe Peter
 * @date November 2020
 * @copyright Developed at SAM, ETH Zurich
 */
#include <lf/assemble/assemble.h>
#include <lf/assemble/coomatrix.h>
#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <memory>

#include "local_assembly.h"

namespace TranspSemiLagr {

/**
 * @brief encorce zero dirichlet boundary conditions
 * @param fe_space shared point to the fe spce on which the problem is defined.
 * @param A reference to the square coefficient matrix in COO format
 * @param b reference to the right-hand-side vector
 */
void enforce_zero_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b);

/**
 * @brief performs a semi lagrangian step according to the update
 * formula 7.3.4.13
 * @param fe_space finite element space on which the problem is solved
 * @param u0_vector nodal values of the solution at the previous time step
 * @param v velocity field (time independent)
 * @param tau time step size
 * @return nodal values of the approximated solution at current time step
 */

/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTOR>
class SemiLagrStep {
 public:
  /* SAM_LISTING_BEGIN_6 */
  SemiLagrStep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
      FUNCTOR v)
#if SOLUTION
      : fe_space_(fe_space),
        A_(fe_space->LocGlobMap().NumDofs(), fe_space->LocGlobMap().NumDofs()),
        Atmp_(fe_space->LocGlobMap().NumDofs(),
              fe_space->LocGlobMap().NumDofs()),
        Up_(fe_space->LocGlobMap().NumDofs()),
        vp_(2, fe_space->LocGlobMap().NumDofs()),
        N_dofs_(fe_space->LocGlobMap().NumDofs())
#else
#endif
  {
#if SOLUTION
    // precalculate the finite element Galerkin matrix $\VA$ for the bilinear
    // form $(u,v) \mapsto \int_\Omega gradu gradv d\Bx$ associated with
    // $-\Delta$. Use \eigen's built-in facilities for the computation of the
    // corresponding element matrices.
    lf::assemble::COOMatrix<double> A_COO(N_dofs_, N_dofs_);
    lf::uscalfe::ReactionDiffusionElementMatrixProvider
        galerkin_element_matrix_provider(
            fe_space_, lf::mesh::utils::MeshFunctionConstant(1.0),
            lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(
        0, fe_space->LocGlobMap(), fe_space->LocGlobMap(),
        galerkin_element_matrix_provider, A_COO);
    Eigen::VectorXd b_dummy = Eigen::VectorXd::Zero(N_dofs_);
    enforce_zero_boundary_conditions(fe_space_, A_COO, b_dummy);

    A_ = A_COO.makeSparse();
    Atmp_ = A_;

    // Precalculate the quadrature weight contributions Up\_.
    // Note that mass\_matrix will be a diagonal matrix
    // because the eval function of LumpedMassElementMatrixProcider always
    // returns diagonal matrices. Could be implemented also without the detaour
    // via a Eigen::SparseMatrix.
    LumpedMassElementMatrixProvider lumped_mass_element_matrix_provider(
        [](const Eigen::Vector2d& /*x*/) { return 1.; });
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs_, N_dofs_);
    lf::assemble::AssembleMatrixLocally(
        0, fe_space->LocGlobMap(), fe_space->LocGlobMap(),
        lumped_mass_element_matrix_provider, mass_matrix);
    Eigen::SparseMatrix<double> mass_sparse = mass_matrix.makeSparse();
    Up_ = mass_sparse.diagonal();

    // precalculate vp\_, v evaluated at each node in the mesh
    for (const auto& node : fe_space->Mesh()->Entities(2)) {
      const lf::geometry::Geometry* geo_ptr = node->Geometry();
      Eigen::Vector2d position = lf::geometry::Corners(*geo_ptr);
      const lf::assemble::size_type idx = fe_space->Mesh()->Index(*node);
      vp_.col(idx) = v(position);
    }
#else
    //====================
    // Your code goes here
    //====================
#endif
  };
  /* SAM_LISTING_END_6 */

  /* SAM_LISTING_BEGIN_7 */
  Eigen::VectorXd step(const Eigen::VectorXd& u0_vector, double tau) {
#if SOLUTION
    // add the scaled diagonal values of Up\_ to the matrix
    for (int i = 0; i < N_dofs_; i++) {
      Atmp_.coeffRef(i, i) = A_.coeff(i, i) + Up_(i) / tau;
    }
    // Warp u0 into a mesh function (required by the Vector provider) \&
    // assemble rhs.
    auto u0_mf = lf::fe::MeshFunctionFE(fe_space_, u0_vector);
    UpwindLagrangianElementVectorProvider vector_provider(
        vp_, tau, fe_space_->Mesh(), u0_mf);

    Eigen::VectorXd b = Eigen::VectorXd::Zero(N_dofs_);
    lf::assemble::AssembleVectorLocally(0, fe_space_->LocGlobMap(),
                                        vector_provider, b);
    // set rhs vector to zero for all boundary nodes to ensure
    // that the Dirichlet boundary conditions are enforced correctly
    lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{
        lf::mesh::utils::flagEntitiesOnBoundary(fe_space_->Mesh(), 2)};
    auto entities = fe_space_->Mesh()->Entities(2);
    for (int i = 0; i < b.size(); i++) {
      if (bd_flags(*entities[i])) {
        b(i) = 0.;
      }
    }
    // solve LSE
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(Atmp_);
    return solver.solve(b);
#else
    //====================
    // Your code goes here
    //====================
    return Eigen::VectorXd::Ones(u0_vector.size());
#endif
  }
  /* SAM_LISTING_END_7 */

 private:
#if SOLUTION
  // Finite Element Space
  std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
  // Galerkin matrix corresponding to the negative Laplacian $a(u,v) \mapsto
  // \int_\Omega gradu gradv d\Bx$ with zero dirichlet boundary condition
  Eigen::SparseMatrix<double> A_;
  // copy of A where the diagonal entries have to be modified in each step
  Eigen::SparseMatrix<double> Atmp_;
  // mass vector where $Up\_(i) = 1/3|U_i|$, where |U_i| represents the sum of
  // the areas of all triangles adjacent to p
  // $=> D(tau) = 1/tau * diag(Up\_)$
  Eigen::VectorXd Up_;
  // v evaluated at each node in the mesh, vp\_.col(i) = v(i)
  Eigen::MatrixXd vp_;
  // number of nodes in mesh
  const lf::uscalfe::size_type N_dofs_;
#else
#endif
};
/* SAM_LISTING_END_1 */

/**
 * @brief approximates the solution to the first model problem specified in the
 * exercise sheet based on N uniform timesteps of the Semi Lagrangian method
 * @param fe_space (linear) finite element space on which the solution is
 * approximated
 * @param u0_vector vector of nodal values of the initial condition
 * @param N number of time steps
 * @param T final time
 */
Eigen::VectorXd solverot(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd u0_vector, int N, double T);

/**
 * @param solves the variational evolution problem specified in the exercise
 * sheet over on time step.
 * @param fe_space finite element space on which the variational evolution
 * problem is solved
 * @param u0_vector nodal values of the solution at the previous time step
 * @param c coefficient function in the variational evolution problem
 * @param tau time step size
 */

/* SAM_LISTING_BEGIN_2 */
template <typename FUNCTOR>
class ReactionStep {
 public:
  ReactionStep(
      std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
      FUNCTOR c)
      : fe_space_(fe_space), c_(c) {
#if SOLUTION
    // Assemble matrix $A_{m,c}$
    LumpedMassElementMatrixProvider mass_element_matrix_provider_c(c_);
    lf::assemble::COOMatrix<double> mass_matrix_c(
        fe_space_->LocGlobMap().NumDofs(), fe_space_->LocGlobMap().NumDofs());
    lf::assemble::AssembleMatrixLocally(
        0, fe_space_->LocGlobMap(), fe_space_->LocGlobMap(),
        mass_element_matrix_provider_c, mass_matrix_c);
    mass_matrix_c_sparse_ = mass_matrix_c.makeSparse();

    // Assemble matrix $A_m$
    LumpedMassElementMatrixProvider mass_element_matrix_provider_1(
        [](Eigen::Vector2d /*x*/) { return 1.0; });
    lf::assemble::COOMatrix<double> mass_matrix_1(
        fe_space_->LocGlobMap().NumDofs(), fe_space_->LocGlobMap().NumDofs());
    lf::assemble::AssembleMatrixLocally(
        0, fe_space_->LocGlobMap(), fe_space_->LocGlobMap(),
        mass_element_matrix_provider_1, mass_matrix_1);
    mass_matrix_1_sparse_ = mass_matrix_1.makeSparse();

#else
    //====================
    // Your code goes here
    //====================
#endif
  };

  Eigen::VectorXd step(const Eigen::VectorXd& u0_vector, double tau) {
#if SOLUTION
    // The explicit midpoint rule computes the solution to $y' = f(y)$ as
    // $y* = y_n + tau/2*f(y_n)$
    // $y_{n+1} =   y_n + tau*f(y*)$

    // In the evolution problem f(y) solves the discretized variational problem
    // $A_m*f(y) = A_{m,c}*y$
    // where $A_m$ is the lumped mass matrix
    // and $A_{m,c}$ is the lumped mass matrix with coefficient function c.

    // initialize sparse solver for mass matrix $A_m$
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(mass_matrix_1_sparse_);

    // evaluate $f(y_n)$
    Eigen::VectorXd k1 = solver.solve(mass_matrix_c_sparse_ * u0_vector);

    // evaluate $f(y*)$
    Eigen::VectorXd k2 =
        solver.solve(mass_matrix_c_sparse_ * (u0_vector + 0.5 * tau * k1));

    // update to $y_{n+1}$
    return u0_vector + tau * k2;
#else
    //====================
    // Your code goes here
    //====================
    return Eigen::VectorXd::Ones(u0_vector.size());
#endif
  }

 private:
  std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
  FUNCTOR c_;
  Eigen::SparseMatrix<double> mass_matrix_c_sparse_;
  Eigen::SparseMatrix<double> mass_matrix_1_sparse_;
};
/* SAM_LISTING_END_2 */

/**
 * @brief approximates the solution to the second model problem specified in the
 * exercise sheet based on N uniform time steps of the Strang-splitting
 * split-step method
 * @param fe_space (linear) finite element space on which the solution is
 * approximated
 * @param u0_vector vector of nodal values of the initial condition
 * @param N number of time steps
 * @param T final time
 */
Eigen::VectorXd solvetrp(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd u0_vector, int N, double T);

void visSLSolution();

void vistrp();

}  // namespace TranspSemiLagr
