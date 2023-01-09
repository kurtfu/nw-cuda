/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/creator.hpp"

#include "cxxopts.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>

/*****************************************************************************/
/*  MODULE FUNCTIONS                                                         */
/*****************************************************************************/

cxxopts::ParseResult parse_program_argumnets(int argc, char const** argv)
{
    cxxopts::Options opts("nw");

    auto type = cxxopts::value<std::string>();

    opts.add_options()("h,help", "Display help");
    opts.add_options()("a,approach", "Specify the approach", type, "APPROACH");
    opts.add_options()("i,input", "Specify the input samples", type, "FILE");
    opts.add_options()("o,output", "Specify the output file", type, "FILE");

    auto args = opts.parse(argc, argv);

    if (args.count("help") != 0)
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

void profile(nw::approach approach, std::ifstream input, std::ofstream output)
{
    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto begin = std::chrono::high_resolution_clock::now();

        auto creator = nw::creator(approach);

        auto nw = creator.create(1, -1, -2);
        auto result = nw->score(nw::input{ref}, nw::input{src});

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        std::cout << "Exec Time: " << time.count() << '\n';
        output << src.length() << ',' << time.count() << ',' << result << '\n';
    }
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

        auto input = args["input"].as<std::string>();
        auto output = args["output"].as<std::string>();

        profile(approach, std::ifstream{input}, std::ofstream{output});
        std::cout << "Testing has been completed!\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
