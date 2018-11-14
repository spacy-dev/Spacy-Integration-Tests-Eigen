#include <Spacy/Adapter/Eigen/Vector.h>
#include <Spacy/Adapter/Eigen/VectorCreator.h>
#include <Spacy/Adapter/Eigen/C2Functional.h>
#include <Spacy/Adapter/Eigen/LinearOperator.h>
#include <Spacy/Adapter/Eigen/LinearSolver.h>
#include <Spacy/linearSolver.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/Cast.h>

#include "Setup.h"

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Rn,C2Functional_Apply)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto f = testFunctional(V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto w = f(v);
  ASSERT_EQ( w , 5. );
}

TEST(Rn,C2Functional_Derivative)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto f = testFunctional(V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto w = f.d1(v)(v);
  ASSERT_EQ( w , 10. );
}

TEST(Rn,C2Functional_SecondDerivative)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto f = testFunctional(V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto w = f.d2(v,v)(v);
  ASSERT_EQ( w , 10. );
}

TEST(Rn,C2Functional_ApplyHessian)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto f = testFunctional(V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto w = f.hessian(v)(v)(v);
  ASSERT_EQ( w , 10. );
}

TEST(Rn,C2Functional_InverseHessian)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto f = testFunctional(V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto A = f.hessian(v).solver();
  auto w = A(v)(v);
  ASSERT_EQ( w , 2.5 );
}
