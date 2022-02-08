#ifndef CLI_VALUE_HPP
#define CLI_VALUE_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "cxxopts.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace cli
{
    using value = std::shared_ptr<cxxopts::Value const>;

    template <typename T>
    cli::value type()
    {
        return cxxopts::value<T>();
    }
}

#endif  // CLI_VALUE_HPP
