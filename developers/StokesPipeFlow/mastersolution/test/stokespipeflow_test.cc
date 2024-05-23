/**
 * @file StokesPipeFlow_test.cc
 * @brief NPDE homework StokesPipeFlow code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../stokespipeflow.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace StokesPipeFlow::test {

TEST(StokesPipeFlow, TaylorHoodElementMatrixProvider) {
  // Populate Galerkin Matrix
  /*
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    lf::io::GmshReader reader(std::move(mesh_factory), "meshes/pipe.msh");

    const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
    const lf::mesh::Mesh &mesh{*mesh_ptr};

    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);

    lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                           {{lf::base::RefEl::kPoint(), 3},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(),0},
                                            {lf::base::RefEl::kQuad(),0}});

    auto elmat_builder = StokesPipeFlow::TaylorHoodElementMatrixProvider();

    std::cout << out << std::endl;*/
}

TEST(StokesPipeFlow, solvePipeFlow) {
  // Populate Galerkin Matrix
  
}

}  // namespace StokesPipeFlow::test
