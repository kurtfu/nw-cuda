#ifndef NW_CREATOR_HPP
#define NW_CREATOR_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"
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

        std::unique_ptr<aligner> create(int match, int miss, int gap);

    private:
        approach type;
    };
}

#endif  // NW_CREATOR_HPP
