#include "nw/cuda.hpp"
#include "catch2/catch.hpp"

TEST_CASE("CUDA score - 1")
{
    std::string ref = "tuvfe";
    std::string src = "kuvaf";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -1);
}

TEST_CASE("CUDA score - 2")
{
    std::string ref = "gattaca";
    std::string src = "gtcgacgca";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -3);
}

TEST_CASE("CUDA score - 3")
{
    std::string ref = "similarity";
    std::string src = "pillar";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -6);
}

TEST_CASE("CUDA fill - 1")
{
    std::string ref = "tuvfe";
    std::string src = "kuvaf";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.fill(ref, src) == -1);
}

TEST_CASE("CUDA fill - 2")
{
    std::string ref = "gattaca";
    std::string src = "gtcgacgca";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.fill(ref, src) == -3);
}

TEST_CASE("CUDA fill - 3")
{
    std::string ref = "similarity";
    std::string src = "pillar";

    nw::cuda nw(1, -1, -2);

    REQUIRE(nw.fill(ref, src) == -6);
}
