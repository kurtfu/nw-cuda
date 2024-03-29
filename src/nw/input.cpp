/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/input.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::input;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

input::input(std::string const& sequence)
    : sequence{' ' + sequence}
{}

char const& input::operator[](std::size_t pos) const
{
    return sequence[pos];
}

std::size_t input::length() const
{
    return sequence.size();
}
