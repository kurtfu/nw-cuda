/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/align.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using profiler::align;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

std::string align::run(nw::input const& ref, nw::input const& src)
{
    auto creator = nw::creator(nw_approach);

    auto nw = creator.create(match, miss, gap);
    std::string alignment = nw->align(ref, src);

    return alignment;
}
