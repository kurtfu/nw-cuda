#ifndef PROFILER_ALIGN_HPP
#define PROFILER_ALIGN_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/base.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace profiler
{
    class align : public base
    {
    public:
        ~align() override = default;

        std::string run(nw::input const& ref, nw::input const& src) override;
    };
}

#endif  // PROFILER_ALIGN_HPP
