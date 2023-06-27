
#ifndef BLENDED_PARAMETRIZATION_H
#define BLENDED_PARAMETRIZATION_H
/**
 * @file blendedparameterization.h
 * @brief NPDE homework BlendedParameterization code
 * @author R. Hiptmair
 * @date January 2018
 * @copyright Developed at SAM, ETH Zurich
 */



#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <limits>


namespace BlendedParameterization {
// fundamental types
using numeric_t = double;
using sparseMatrix_t = Eigen::SparseMatrix<numeric_t>;
using matrix_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic>;
using vector_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, 1>;
using coord_t = Eigen::Matrix<numeric_t, 2, 1>;

// virtual base class describing a curve
class Curve {
 public:
  virtual coord_t operator()(double parameter) const = 0;
  virtual coord_t derivative(double parameter) const = 0;
};

// sample child class of 'Curve': straight line segment
class Segment : public Curve {
 public:
  // Constructor sets endpoints
  Segment(const coord_t& a0, const coord_t& a1) : a0_(a0), a1_(a1) {}
  // Accessing points on the line segment: affine parameterization
  virtual coord_t operator()(double parameter) const override final {
    return (1. - parameter) * a0_ + parameter * a1_;
  }
  // Constant derivative for a line segment
  virtual coord_t derivative(double parameter) const override final {
    return a1_ - a0_;
  }

 private:
  coord_t a0_;  // vertex 1
  coord_t a1_;  // vertex 2
};

matrix_t evalBlendLocMat(const Curve& gamma01, const Curve& gamma12,
                         const Curve& gamma20);

}  // namespace BlendedParameterization

#endif //BLENDED_PARAMETRIZATION_H
