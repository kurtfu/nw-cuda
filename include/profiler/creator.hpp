#ifndef PROFILER_CREATOR_HPP
#define PROFILER_CREATOR_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/base.hpp"
#include <memory>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace profiler
{
    enum class approach
    {
        align,
        score
    };

    class creator
    {
    public:
        static std::unique_ptr<base> create(approach type);
    };
}

#endif  // PROFILER_CREATOR_HPP
