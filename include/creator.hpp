#ifndef NW_CREATOR_HPP
#define NW_CREATOR_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "aligner.hpp"
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
        creator() = delete;
        static std::unique_ptr<aligner> create(algo type, int match, int miss, int gap);
    };
}

#endif  // NW_CREATOR_HPP
