/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "cli/parser.hpp"
#include "nw/creator.hpp"

/*****************************************************************************/
/*  TYPE ALIASES                                                             */
/*****************************************************************************/

using test_fn = int (nw::aligner::*)(std::string const&, std::string const&);

/*****************************************************************************/
/*  MODULE VARIABLES                                                         */
/*****************************************************************************/

static std::unordered_map<std::string, nw::approach> approach = {
    {"cuda",   nw::approach::cuda  },
    {"serial", nw::approach::serial},
};

static std::unordered_map<std::string, test_fn> test = {
    {"fill",  &nw::aligner::fill },
    {"score", &nw::aligner::score}
};

/*****************************************************************************/
/*  MODULE FUNCTIONS                                                         */
/*****************************************************************************/

cli::result parse_arguments(cli::parser& parser, int argc, char const* argv[])
{
    auto result = parser.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << parser.help() << '\n';
        std::exit(EXIT_SUCCESS);
    }

    auto func = result["test"].as<std::string>();
    auto type = result["approach"].as<std::string>();

    if (test.find(func) == test.end())
    {
        std::cerr << func + " is not a valid test function\n";
        std::exit(EXIT_FAILURE);
    }

    if (approach.find(type) == approach.end())
    {
        std::cerr << type + " is not a valid approach\n";
        std::exit(EXIT_FAILURE);
    }

    auto samples = result["input"].as<std::string>();

    if (std::ifstream(samples).good() == false)
    {
        std::cerr << '\'' + samples + "\' is not existing\n";
        std::exit(EXIT_FAILURE);
    }

    return result;
}

void process_results(cli::result const& result)
{
    constexpr int match = 1;
    constexpr int miss  = -1;
    constexpr int gap   = -2;

    auto samples = result["input"].as<std::string>();
    auto log     = result["output"].as<std::string>();

    std::ifstream input(samples);
    std::ofstream output(log);

    auto func = result["test"].as<std::string>();
    auto type = result["approach"].as<std::string>();

    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto creator = nw::creator(approach[type]);

        auto begin = std::chrono::high_resolution_clock::now();

        auto nw    = creator.create(match, miss, gap);
        int  score = std::invoke(test[func], *nw, ref, src);

        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        std::cout << "Exec Time: " << elapsed.count() << '\n';
        output << src.size() << ',' << score << ',' << elapsed.count() << '\n';
    }

    std::cout << "Testing has been completed!\n";
}

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    cli::parser parser("nw");

    auto type = cli::type<std::string>();

    parser.add_option("a,approach", "Specify the approach", type, "APPROACH");
    parser.add_option("h,help", "Display help");
    parser.add_option("i,input", "Specify the input samples", type, "FILE");
    parser.add_option("o,output", "Specify the output file", type, "FILE");
    parser.add_option("t,test", "Specify the test function", type, "FUNCTION");

    try
    {
        auto result = parse_arguments(parser, argc, argv);
        process_results(result);
    }
    catch (std::exception const& ex)
    {
        std::cerr << ex.what() << '\n' + parser.help() << '\n';
    }

    return 0;
}
