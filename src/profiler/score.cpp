/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/score.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using profiler::score;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

std::string score::run(nw::input const& ref, nw::input const& src)
{
    auto creator = nw::creator(nw_approach);

    auto nw = creator.create(match, miss, gap);
    int score = nw->score(ref, src);

    return std::to_string(score);
}
