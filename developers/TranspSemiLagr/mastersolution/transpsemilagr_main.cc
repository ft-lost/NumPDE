/**
 * @file transpsemilagr_main.cc
 * @brief NPDE homework TranspSemiLagr Main file
 * @author Philippe Peter
 * @date November 2020
 * @copyright Developed at SAM, ETH Zurich
 */

#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <filesystem>
#include <memory>

#include "transpsemilagr.h"

int main() {
  TranspSemiLagr::visSLSolution();
  TranspSemiLagr::vistrp();

  return 0;
}
