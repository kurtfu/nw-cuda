/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <iostream>
#include <unordered_map>

#include "cxxopts.hpp"

#include "nw/creator.hpp"
#include "profiler/creator.hpp"

/*****************************************************************************/
/*  MODULE FUNCTIONS                                                         */
/*****************************************************************************/

cxxopts::ParseResult parse_program_argumnets(int argc, char const* argv[])
{
    cxxopts::Options opts("nw");

    auto type = cxxopts::value<std::string>();

    opts.add_options()("a,approach", "Specify the approach", type, "APPROACH");
    opts.add_options()("h,help", "Display help");
    opts.add_options()("i,input", "Specify the input samples", type, "FILE");
    opts.add_options()("o,output", "Specify the output file", type, "FILE");
    opts.add_options()("t,test", "Specify the test method", type, "METHOD");

    auto args = opts.parse(argc, argv);

    if (args.count("help"))
    {
        std::cout << opts.help() << '\n';
        std::exit(EXIT_SUCCESS);
    }

    return args;
}

template <>
nw::approach const& cxxopts::OptionValue::as<nw::approach>() const
{
    static std::unordered_map<std::string, nw::approach> opts = {
        {"serial", nw::approach::serial},
        {"cuda",   nw::approach::cuda  },
    };

    auto arg = this->as<std::string>();

    if (opts.find(arg) == opts.end())
    {
        throw std::runtime_error("\'" + arg + "\' is not a valid approach");
    }

    return opts[arg];
}

template <>
profiler::approach const& cxxopts::OptionValue::as<profiler::approach>() const
{
    static std::unordered_map<std::string, profiler::approach> opts = {
        {"align", profiler::approach::align},
        {"score", profiler::approach::score},
    };

    auto arg = this->as<std::string>();

    if (opts.find(arg) == opts.end())
    {
        throw std::runtime_error("\'" + arg + "\' is not a valid profiler");
    }

    return opts[arg];
}

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    try
    {
        auto args = parse_program_argumnets(argc, argv);

        auto approach = args["approach"].as<nw::approach>();
        auto test = args["test"].as<profiler::approach>();

        auto samples = args["input"].as<std::string>();
        auto log = args["output"].as<std::string>();

        auto profiler = profiler::creator::create(test);

        profiler->assign_scoring_coefficients(1, -1, -2);
        profiler->assign_nw_approach(approach);

        profiler->attach_input(samples);
        profiler->attach_output(log);

        profiler->profile_samples();
        std::cout << "Testing has been completed!\n";
    }
    catch (std::exception const& ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return 0;
}
