set(SOURCES
        ${DIR}/blendedparameterization_main.cc
        ${DIR}/blendedparameterization.cc
        ${DIR}/blendedparameterization.h
        ${DIR}/MeshTriangleUnitSquareEigen.hpp
        )
set(LIBRARIES Eigen3::Eigen LF::lf.base LF::lf.mesh LF::lf.mesh.test_utils LF::lf.quad LF::lf.assemble LF::lf.refinement LF::lf.uscalfe)
