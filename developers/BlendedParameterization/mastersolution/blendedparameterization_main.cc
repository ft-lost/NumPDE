/**
 * @ file blendedparameterization_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author R. Hiptmair
 * @ date January 2018
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "blendedparameterization.h"

int main(int /*argc*/, char** /*argv*/) {
  // create a triangular grid on unit square
  int N = 100;
  BlendedParameterization::matrix_t elements = generateMesh(N);
  // ---

  // iterate over all triangles
  for (int i = 0; i < elements.rows(); ++i) {
    BlendedParameterization::coord_t a0 = elements.row(i).head(2);
    BlendedParameterization::coord_t a1 = elements.row(i).segment(2, 2);
    BlendedParameterization::coord_t a2 = elements.row(i).tail(2);

    BlendedParameterization::Segment gamma01(a0, a1);
    BlendedParameterization::Segment gamma12(a1, a2);
    BlendedParameterization::Segment gamma20(a2, a0);

    BlendedParameterization::matrix_t lclMat =
        BlendedParameterization::evalBlendLocMat(gamma01, gamma12, gamma20);
  }

  return 0;
}
