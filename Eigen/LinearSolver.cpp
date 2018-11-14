#include <Spacy/Adapter/Eigen/LinearSolver.h>
#include <Spacy/Adapter/Eigen/Vector.h>
#include <Spacy/Adapter/Eigen/VectorCreator.h>
#include <Spacy/vector.hh>
#include <Spacy/vectorSpace.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/Cast.h>

#include <Mock/Norm.h>
#include <Mock/Vector.h>

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Eigen, LinearSolver)
{
    const auto dim = 2;
    ::Eigen::MatrixXd A(dim, dim);
    A(0,0) = 2;
    A(0,1) = 0;
    A(1,1) = 1;
    A(1,0) = 0;

    ::Eigen::VectorXd v(dim);
    v(0) = 1;
    v(1) = 1;

    const auto V = Rn::makeHilbertSpace(dim);
    const auto w = Rn::Vector(v, V);
    Rn::LinearSolver solver(A, V);

    const auto x = solver(w);
    auto& x_eigen = cast_ref<Rn::Vector>(x).get();
    EXPECT_DOUBLE_EQ(x_eigen(0), 0.5);
    EXPECT_DOUBLE_EQ(x_eigen(1), 1);
}
