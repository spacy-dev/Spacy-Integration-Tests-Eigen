#include <Spacy/Adapter/Eigen/LinearSolver.h>
#include <Spacy/vector.hh>
#include <Spacy/vectorSpace.hh>
#include <Spacy/Util/Cast.h>

#include <Mock/Norm.h>
#include <Mock/Vector.h>

#include <gtest/gtest.h>

using namespace Spacy;

TEST(Eigen, LinearSolver)
{
    ::Eigen::MatrixXd A(2,2);
    A(0,0) = 2;
    A(1,1) = 1;

    ::Eigen::VectorXd v(2);
    v(0) = 1;
    v(1) = 1;

    const auto V = VectorSpace(Mock::VectorCreator(), Mock::Norm());
    const auto w = Rn::Vector(v, V);
    Rn::LinearSolver solver(A, V);

    const auto x = solver(w);
    auto& x_eigen = cast_ref<RN::Vector>(x).get();
    EXPECT_DOUBLE_EQ(x_eigen(0), 0.5);
    EXPECT_DOUBLE_EQ(x_eigen(1), 1);
}
