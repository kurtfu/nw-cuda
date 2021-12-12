#ifndef NW_ALIGNER_HPP
#define NW_ALIGNER_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <string>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class aligner
    {
    public:
        virtual int score(std::string const& ref, std::string const& src) = 0;

    protected:
        int match;
        int miss;
        int gap;
    };
}

#endif  // NW_ALIGNER_HPP
