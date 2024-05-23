/**
 * @file stokespipeflow.cc
 * @brief NPDE homework StokesPipeFlow code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "stokespipeflow.h"

#include <lf/assemble/assembly_types.h>
#include <lf/assemble/coomatrix.h>
#include <lf/mesh/entity.h>

namespace StokesPipeFlow {

TaylorHoodElementMatrixProvider::ElemMat TaylorHoodElementMatrixProvider::Eval(
    const lf::mesh::Entity &cell) {
  AK_.setZero();
  std::cerr << "Not yet implemented!\n";
  return AK_;
}

lf::assemble::COOMatrix<double> buildTaylorHoodGalerkinMatrix(
    const lf::assemble::DofHandler &dofh) {
  // Total number of FE d.o.f.s without Lagrangian multiplier
  lf::assemble::size_type n = dofh.NumDofs();
  // Full Galerkin matrix in triplet format taking into account the zero mean
  // constraint on the pressure.
  lf::assemble::COOMatrix<double> A(n+1,n+1);

  return A;
}

}  // namespace StokesPipeFlow
