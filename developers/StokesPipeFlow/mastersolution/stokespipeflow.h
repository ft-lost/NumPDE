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

}  // namespace StokesPipeFlow

#endif
