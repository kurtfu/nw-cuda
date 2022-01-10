#ifndef NW_CREATOR_HPP
#define NW_CREATOR_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "aligner.hpp"

#include <functional>
#include <memory>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    enum class algo
    {
        serial,
        cuda
    };

    class creator
    {
    public:
        creator(algo type);

        std::function<std::unique_ptr<aligner>(int match, int miss, int gap)> create;
    };
}

#endif  // NW_CREATOR_HPP
