/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/creator.hpp"

#include "profiler/align.hpp"
#include "profiler/score.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using profiler::creator;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

std::unique_ptr<profiler::base> creator::create(profiler::approach type)
{
    switch (type)
    {
        case approach::align:
            return std::make_unique<profiler::align>();
        case approach::score:
            return std::make_unique<profiler::score>();
        default:
            return nullptr;
    }
}
