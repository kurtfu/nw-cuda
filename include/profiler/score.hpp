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
        score() = default;

        score(score const& that) = delete;
        score(score&& that) = delete;

        ~score() override = default;

        score& operator=(score const& that) = delete;
        score& operator=(score&& that) = delete;

        std::string run(nw::input const& ref, nw::input const& src) override;
    };
}

#endif  // PROFILER_SCORE_HPP
