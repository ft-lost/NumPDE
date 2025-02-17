/**
 * @file fvmshallowwater.cc
 * @brief NPDE homework FVMShallowWater code
 * @author Ralf Hiptmair
 * @date June 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#include "fvmshallowwater.h"

#include <cmath>
#include <limits>

namespace FVMShallowWater {


/* SAM_LISTING_BEGIN_1 */
Eigen::Vector2d numfluxLFSWE(Eigen::Vector2d v, Eigen::Vector2d w) {
  assert((v[0] > 0.0) && "v-height must be positive!");
  assert((w[0] > 0.0) && "w-height must be positive!");
  Eigen::Vector2d nfLF{};
  /* **************************************************
     Your code here
     ************************************************** */
  return nfLF;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_3 */
Eigen::Vector2d numfluxHLLESWE(Eigen::Vector2d v, Eigen::Vector2d w) {
  assert((v[0] > 0.0) && "v-height must be positive!");
  assert((w[0] > 0.0) && "w-height must be positive!");
  Eigen::Vector2d nfHLLE{};
  /* **************************************************
     Your code here
     ************************************************** */
  return nfHLLE;
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
/* SAM_LISTING_END_4 */

/* SAM_LISTING_BEGIN_5 */
/* SAM_LISTING_END_5 */

/* SAM_LISTING_BEGIN_6 */
bool isPhysicalTwoShockSolution(Eigen::Vector2d ul, Eigen::Vector2d us,
                                Eigen::Vector2d ur) {
  return true;
}
/* SAM_LISTING_END_6 */

}  // namespace FVMShallowWater
