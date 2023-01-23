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
        explicit input(std::string const& sequence);

        char const& operator[](std::size_t pos) const;
        [[nodiscard]] std::size_t length() const;

    private:
        std::string sequence;
    };
}

#endif  // NW_INPUT_HPP
