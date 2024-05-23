/**
 * @file stokespipeflow.h
 * @brief NPDE homework StokesPipeFlow code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#ifndef StokesPipeFlow_H_
#define StokesPipeFlow_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <lf/assemble/assemble.h>
#include <lf/assemble/coomatrix.h>
#include <lf/assemble/dofhandler.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace StokesPipeFlow {
/**
 * @brief Element matrix provider for Taylor-Hood Stokes FEM
 */
/* SAM_LISTING_BEGIN_1 */
class TaylorHoodElementMatrixProvider {
 public:
  using ElemMat = Eigen::Matrix<double, 15, 15>;
  TaylorHoodElementMatrixProvider(const TaylorHoodElementMatrixProvider &) =
      delete;
  TaylorHoodElementMatrixProvider(TaylorHoodElementMatrixProvider &&) noexcept =
      default;
  TaylorHoodElementMatrixProvider &operator=(
      const TaylorHoodElementMatrixProvider &) = delete;
  TaylorHoodElementMatrixProvider &operator=(
      TaylorHoodElementMatrixProvider &&) = delete;
  TaylorHoodElementMatrixProvider() = default;
  virtual ~TaylorHoodElementMatrixProvider() = default;
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  [[nodiscard]] ElemMat Eval(const lf::mesh::Entity &cell);

 private:
  ElemMat AK_;
};
/* SAM_LISTING_END_1 */

/**
 * @brief Assembly of full Galerkin matrix in triplet format
 *
 * @param dofh DofHandler object for all FE spaces
 */
lf::assemble::COOMatrix<double> buildTaylorHoodGalerkinMatrix(
    const lf::assemble::DofHandler &dofh);

/**
 * @brief Taylor-Hood FE solultion of pipe flow problem
 *
 * @tparam functor type taking a 2-vector and returning a 2-vector
 * @param dofh DofHandler object for all FE spaces
 * @param g functor providing Dirchlet boundary data
 */
template <typename gFunctor>
Eigen::VectorXd solvePipeFlow(const lf::assemble::DofHandler &dofh,
                              gFunctor &&g) {
  // Total number of FE d.o.f.s without Lagrangian multiplier
  lf::assemble::size_type n = dofh.NumDofs();
  // Vector of all basis expansion coefficients of Taylor-Hood finite element
  // solution of pipe flow problem. This covers both velocity and pressure.
  Eigen::VectorXd dofvec{n};

  return dofvec;
}

}  // namespace StokesPipeFlow

#endif
