#include "nw/serial.hpp"
#include "catch2/catch.hpp"

TEST_CASE("Serial score - 1")
{
    std::string ref = "tuvfe";
    std::string src = "kuvaf";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -1);
}

TEST_CASE("Serial score - 2")
{
    std::string ref = "gattaca";
    std::string src = "gtcgacgca";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -3);
}

TEST_CASE("Serial score - 3")
{
    std::string ref = "similarity";
    std::string src = "pillar";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.score(ref, src) == -6);
}

TEST_CASE("Serial align - 1")
{
    std::string ref = "tuvfe";
    std::string src = "kuvaf";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.align(ref, src) == "!**!!");
}

TEST_CASE("Serial align - 2")
{
    std::string ref = "gattaca";
    std::string src = "gtcgacgca";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.align(ref, src) == "*!!!**--*");
}

TEST_CASE("Serial align - 3")
{
    std::string ref = "similarity";
    std::string src = "pillar";

    nw::serial nw(1, -1, -2);

    REQUIRE(nw.align(ref, src) == "!*!-***---");
}