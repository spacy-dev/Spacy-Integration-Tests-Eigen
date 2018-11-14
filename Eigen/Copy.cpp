#include <Spacy/Adapter/Eigen/Copy.h>
#include <Spacy/Adapter/Eigen/VectorCreator.h>

#include <MockSetup.h>

#include "Setup.h"

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Rn,GetSize)
{
  auto V = Rn::makeHilbertSpace(testDim());
  auto x = zero(V);
  ASSERT_EQ( Rn::getSize(x) , testDim() );
}

TEST(Rn,GetSizeOfProductSpace)
{
  auto V = Rn::makeHilbertSpace( {testDim(),testDim(),testDim()} );
  auto x = zero(V);
  ASSERT_EQ( Rn::getSize(x) , 3*testDim() );
}

TEST(Rn,GetSizeOfPrimalDualProductSpace)
{
  auto V = Rn::makeHilbertSpace( {testDim(),testDim(),testDim()} , {1,2} , {0} );
  auto x = zero(V);
  ASSERT_EQ( Rn::getSize(x) , 3*testDim() );
}
