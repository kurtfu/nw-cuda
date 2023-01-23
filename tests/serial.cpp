/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <catch2/catch_test_macros.hpp>

#include "nw/serial.hpp"

/*****************************************************************************/
/*  TEST CASES                                                               */
/*****************************************************************************/

TEST_CASE("Serial score")
{
    nw::serial algo(1, -1, -2);

    SECTION("Test with the same size")
    {
        std::string const ref = "tuvfe";
        std::string const src = "kuvaf";

        REQUIRE(algo.score(nw::input{ref}, nw::input{src}) == -1);
    }

    SECTION("Test with a longer source")
    {
        std::string const ref = "gattaca";
        std::string const src = "gtcgacgca";

        REQUIRE(algo.score(nw::input{ref}, nw::input{src}) == -3);
    }

    SECTION("Test with longer reference")
    {
        std::string const ref = "similarity";
        std::string const src = "pillar";

        REQUIRE(algo.score(nw::input{ref}, nw::input{src}) == -6);
    }
}

TEST_CASE("Serial align")
{
    nw::serial algo(1, -1, -2);

    SECTION("Test with the same size")
    {
        std::string const ref = "tuvfe";
        std::string const src = "kuvaf";

        REQUIRE(algo.align(nw::input{ref}, nw::input{src}) == "!**!!");
    }

    SECTION("Test with a longer source")
    {
        std::string const ref = "gattaca";
        std::string const src = "gtcgacgca";

        REQUIRE(algo.align(nw::input{ref}, nw::input{src}) == "*!!!**--*");
    }

    SECTION("Test with longer reference")
    {
        std::string const ref = "similarity";
        std::string const src = "pillar";

        REQUIRE(algo.align(nw::input{ref}, nw::input{src}) == "!*!-***---");
    }
}
