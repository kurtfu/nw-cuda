#ifndef NW_INPUT_HPP
#define NW_INPUT_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <string>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class input
    {
    public:
        input(std::string const& sequence);

        char const& operator[](std::size_t) const;
        std::size_t length() const;

    private:
        std::string sequence;
    };
}

#endif  // NW_INPUT_HPP
