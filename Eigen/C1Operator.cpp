#include <Spacy/Adapter/Eigen/Vector.h>
#include <Spacy/Adapter/Eigen/VectorCreator.h>
#include <Spacy/Adapter/Eigen/C1Operator.h>
#include <Spacy/Adapter/Eigen/LinearOperator.h>
#include <Spacy/vector.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/Cast.h>

#include "Setup.h"

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Rn,C1Operator_Apply)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto A = testOperator(V,V);
  auto v = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  auto w = A(v);
  const Rn::Vector& w_ = cast_ref<Rn::Vector>(w);
  EXPECT_EQ( w_.get()[0] , 1.1*testVector()[0] );
  EXPECT_EQ( w_.get()[1] , 1.1*testVector()[1] );
}

TEST(Rn,C1Operator_Derivative)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto A = testOperator(V,V);
  auto v = zero(V), w = zero(V);
  cast_ref<Rn::Vector>(v) = testVector();
  w = 2*v;
  auto x = A.d1(v,w);
  const Rn::Vector& w_ = cast_ref<Rn::Vector>(w);
  const Rn::Vector& x_ = cast_ref<Rn::Vector>(x);
  EXPECT_EQ( x_.get()[0] , 1.1*w_.get()[0] );
  EXPECT_EQ( x_.get()[1] , 1.1*w_.get()[1] );
}

TEST(Rn,C1Operator_Linearization)
{
  auto space = Rn::makeHilbertSpace(testDim());
  auto A = testOperator(space,space);
  auto v = zero(space);
  cast_ref<Rn::Vector>(v) = testVector();
  auto dA = A.linearization(v);
  auto w = dA(v);
  const Rn::Vector& w_ = cast_ref<Rn::Vector>(w);
  EXPECT_EQ( w_.get()[0] , 1.1*testVector()[0] );
  EXPECT_EQ( w_.get()[1] , 1.1*testVector()[1] );
}
