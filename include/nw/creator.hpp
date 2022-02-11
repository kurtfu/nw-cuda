#ifndef NW_CREATOR_HPP
#define NW_CREATOR_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

#include <functional>
#include <memory>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    enum class approach
    {
        serial,
        cuda
    };

    class creator
    {
    public:
        creator(approach type);

        std::function<std::unique_ptr<aligner>(int match, int miss, int gap)> create;
    };
}

#endif  // NW_CREATOR_HPP
