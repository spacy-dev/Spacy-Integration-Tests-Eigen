#include <gtest.hh>

#include <Spacy/Adapter/Eigen/vector.hh>
#include <Spacy/Adapter/Eigen/vectorCreator.hh>
#include <mockSetup.hh>

#include "setup.hh"

using namespace Spacy;

TEST(Rn,CreateFromEigen_VectorXd)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacyVector(vec,space);

  ASSERT_EQ( spacyVector.get()[0] , 1.);
  ASSERT_EQ( spacyVector.get()[1] , 2.);
}

TEST(Rn,AssignFromEigen_VectorXd)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(0.*vec,space);

  ASSERT_EQ( spacy_v.get()[0] , 0.);
  ASSERT_EQ( spacy_v.get()[1] , 0.);

  spacy_v = vec;
  ASSERT_EQ( spacy_v.get()[0] , 1.);
  ASSERT_EQ( spacy_v.get()[1] , 2.);
}

TEST(Rn,AddVectors)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(vec,space);

  spacy_v += spacy_v;
  ASSERT_EQ( spacy_v.get()[0] , 2.);
  ASSERT_EQ( spacy_v.get()[1] , 4.);
}

TEST(Rn,SubtractVectors)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(vec,space);

  spacy_v -= spacy_v;
  ASSERT_EQ( spacy_v.get()[0] , 0.);
  ASSERT_EQ( spacy_v.get()[1] , 0.);
}

TEST(Rn,MultiplyVectorWithScalar)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(vec,space);

  spacy_v *= 3;
  ASSERT_EQ( spacy_v.get()[0] , 3.);
  ASSERT_EQ( spacy_v.get()[1] , 6.);
}

TEST(Rn,ApplyAsDual)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(vec,space);

  auto dualPairing = spacy_v(spacy_v);
  ASSERT_EQ( dualPairing , 5.);
}

TEST(Rn,Negation)
{
  auto vec = testVector();

  auto space = createMockBanachSpace();
  Spacy::Rn::Vector spacy_v(vec,space);

  auto spacy_w = -spacy_v;
  ASSERT_EQ( spacy_w.get()[0] , -1.);
  ASSERT_EQ( spacy_w.get()[1] , -2.);
}

TEST(Rn,Comparison)
{
  auto vec = testVector();
  vec[1] = 0;

  auto space = Spacy::Rn::makeHilbertSpace(testDim());
  Spacy::Rn::Vector spacy_v(vec,space), spacy_w(vec,space);

  ASSERT_TRUE( spacy_v == spacy_w );

  space.setEps(1e-5);
  spacy_w.get()[0] = 1 - 5e-6;

  ASSERT_TRUE( spacy_v == spacy_w );
  spacy_w.get()[0] = 1 - 1.1e-5;
  ASSERT_FALSE( spacy_v == spacy_w );
}
