/**
 * @file stokespipeflow.cc
 * @brief NPDE homework StokesPipeFlow code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "stokespipeflow.h"

#include <lf/mesh/entity.h>

namespace StokesPipeFlow {

TaylorHoodElementMatrixProvider::ElemMat TaylorHoodElementMatrixProvider::Eval(
    const lf::mesh::Entity &cell) {
  AK_.setZero();
  return AK_;
}

}  // namespace StokesPipeFlow
