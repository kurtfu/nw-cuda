/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "parser.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using cli::parser;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

parser::parser(std::string const& program, std::string const& desc)
    : options{program, desc}
{}

void parser::add_option(std::string const& opts,
                        std::string const& desc,
                        cli::value const& type,
                        std::string const& usage)
{
    options.add_options()(opts, desc, type, usage);
}

std::string parser::help()
{
    return options.help();
}

cli::result parser::parse(int argc, char const* argv[])
{
    return options.parse(argc, argv);
}
