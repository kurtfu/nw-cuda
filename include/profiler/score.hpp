#ifndef PROFILER_SCORE_HPP
#define PROFILER_SCORE_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/base.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace profiler
{
    class score : public base
    {
    public:
        ~score() = default;

        std::string run(nw::input const& ref, nw::input const& src) override;
    };
}

#endif  // PROFILER_SCORE_HPP
