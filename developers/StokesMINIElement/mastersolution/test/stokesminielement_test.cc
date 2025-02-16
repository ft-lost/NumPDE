/**
 * @file StokesMINIElement_test.cc
 * @brief NPDE homework StokesMINIElement code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../stokesminielement.h"

#include <gtest/gtest.h>
#include <lf/mesh/test_utils/test_meshes.h>

#include <Eigen/Core>

/* Test in the google testing framework

  The following assertions are available, syntax
  EXPECT_XX( ....) << [anything that can be givne to std::cerr]

  EXPECT_EQ(val1, val2)
  EXPECT_NEAR(val1, val2, abs_error) -> should be used for numerical results!
  EXPECT_NE(val1, val2)
  EXPECT_TRUE(condition)
  EXPECT_FALSE(condition)
  EXPECT_GE(val1, val2)
  EXPECT_LE(val1, val2)
  EXPECT_GT(val1, val2)
  EXPECT_LT(val1, val2)
  EXPECT_STREQ(str1,str2)
  EXPECT_STRNE(str1,str2)
  EXPECT_STRCASEEQ(str1,str2)
  EXPECT_STRCASENE(str1,str2)

  "EXPECT" can be replaced with "ASSERT" when you want to program to terminate,
 if the assertion is violated.
 */

namespace StokesMINIElement::test {

TEST(StokesMINIElement, SUBPROBLEM_G) {
  // Define a pre-computed element matrix
  Eigen::MatrixXd exact_elem(11, 11);
  exact_elem << 0.416667, 0, -0.0555556, -0.166667, 0, -0.0555556, -0.25, 0,
      -0.0555556, 0, 0, 0, 0.416667, -0.0277778, 0, -0.166667, -0.0277778, 0,
      -0.25, -0.0277778, 0, 0, -0.0555556, -0.0277778, 0, 0.0555556, -0.0555556,
      0, 0, 0.0833333, 0, 0.00277778, 0.00138889, -0.166667, 0, 0.0555556,
      0.666667, 0, 0.0555556, -0.5, 0, 0.0555556, 0, 0, 0, -0.166667,
      -0.0555556, 0, 0.666667, -0.0555556, 0, -0.5, -0.0555556, 0, 0,
      -0.0555556, -0.0277778, 0, 0.0555556, -0.0555556, 0, 0, 0.0833333, 0,
      -0.00277778, 0.00277778, -0.25, 0, 0, -0.5, 0, 0, 0.75, 0, 0, 0, 0, 0,
      -0.25, 0.0833333, 0, -0.5, 0.0833333, 0, 0.75, 0.0833333, 0, 0,
      -0.0555556, -0.0277778, 0, 0.0555556, -0.0555556, 0, 0, 0.0833333, 0, -0,
      -0.00416667, 0, 0, 0.00277778, 0, 0, -0.00277778, 0, 0, -0, 0.0101852, 0,
      0, 0, 0.00138889, 0, 0, 0.00277778, 0, 0, -0.00416667, 0, 0.0101852;

  // Generate the mesh
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  auto cells = mesh_ptr->Entities(0);

  // Compute the element matrix for the first cell
  StokesMINIElement::MINIElementMatrixProvider MINIemp;
  auto elem = MINIemp.Eval(*cells[0]);

  // Check if the solution is correct
  ASSERT_EQ(elem.cols(), 11);
  ASSERT_EQ(elem.rows(), 11);
  ASSERT_NEAR((elem - exact_elem).norm(), 0., 1e-4);
}

}  // namespace StokesMINIElement::test
