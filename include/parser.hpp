#ifndef CLI_PARSER_HPP
#define CLI_PARSER_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "result.hpp"
#include "value.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace cli
{
    class parser
    {
    public:
        parser(std::string const& program, std::string const& desc = "");
        ~parser() = default;

        void add_option(std::string const& opts,
                        std::string const& desc,
                        cli::value const& type = cli::type<bool>(),
                        std::string const& usage = "<arg>");

        std::string help();
        cli::result parse(int argc, char const* argv[]);

    private:
        cxxopts::Options options;
    };
}

#endif  // CLI_PARSER_HPP
