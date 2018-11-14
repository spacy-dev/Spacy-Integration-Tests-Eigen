#include <Spacy/Adapter/Eigen/Vector.h>
#include <Spacy/Adapter/Eigen/VectorCreator.h>
#include <Spacy/Adapter/Eigen/LinearOperator.h>
#include <Spacy/Adapter/Eigen/LinearSolver.h>
#include <Spacy/vector.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/cast.hh>

#include "Setup.h"

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Rn,LinearOperator_Apply)
{
  auto space = Spacy::Rn::makeHilbertSpace(testDim());
  auto A = Spacy::Rn::LinearOperator(testMatrix(),space,space,space);
  auto v = Spacy::Rn::Vector( testVector() , space );

  auto w = A(v);
  ASSERT_EQ( cast_ref<Spacy::Rn::Vector>(w).get()[0] ,  1. );
  ASSERT_EQ( cast_ref<Spacy::Rn::Vector>(w).get()[1] ,  5. );
}


TEST(Rn,LinearOperator_ApplySolver)
{
  auto space = Spacy::Rn::makeHilbertSpace(testDim());
  auto A = Spacy::Rn::LinearOperator(testMatrix(),space,space,space);
  auto v = zero(space);
  cast_ref<Spacy::Rn::Vector>(v) = testVector();

  auto w = A.solver()(v);
  ASSERT_EQ( cast_ref<Spacy::Rn::Vector>(w).get()[0] ,  1. );
  ASSERT_EQ( cast_ref<Spacy::Rn::Vector>(w).get()[1] ,  .5 );
}
