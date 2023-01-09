/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <catch2/catch_test_macros.hpp>

#include "nw/cuda.hpp"

/*****************************************************************************/
/*  TEST CASES                                                               */
/*****************************************************************************/

TEST_CASE("CUDA score - 1")
{
    std::string const ref = "tuvfe";
    std::string const src = "kuvaf";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(nw::input{ref}, nw::input{src}) == -1);
}

TEST_CASE("CUDA score - 2")
{
    std::string const ref = "gattaca";
    std::string const src = "gtcgacgca";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(nw::input{ref}, nw::input{src}) == -3);
}

TEST_CASE("CUDA score - 3")
{
    std::string const ref = "similarity";
    std::string const src = "pillar";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(nw::input{ref}, nw::input{src}) == -6);
}

TEST_CASE("CUDA align - 1")
{
    std::string const ref = "tuvfe";
    std::string const src = "kuvaf";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.align(nw::input{ref}, nw::input{src}) == "!**!!");
}

TEST_CASE("CUDA align - 2")
{
    std::string const ref = "gattaca";
    std::string const src = "gtcgacgca";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.align(nw::input{ref}, nw::input{src}) == "*!!!**--*");
}

TEST_CASE("CUDA align - 3")
{
    std::string const ref = "similarity";
    std::string const src = "pillar";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.align(nw::input{ref}, nw::input{src}) == "!*!-***---");
}
