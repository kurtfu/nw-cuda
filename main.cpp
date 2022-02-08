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

constexpr int match = 1;
constexpr int miss  = -1;
constexpr int gap   = -2;

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    std::unordered_map<std::string, nw::algo> algo = {
        {"cuda",   nw::algo::cuda  },
        {"serial", nw::algo::serial},
    };

    std::unordered_map<std::string, test_fn> test = {
        {"fill",  &nw::aligner::fill },
        {"score", &nw::aligner::score}
    };

    cli::parser parser("nw", " - Needleman-Wunsch example program");

    parser.add_option("a,algo", "Specify the algorithm", cli::type<std::string>());
    parser.add_option("h,help", "Display help");
    parser.add_option("i,input", "Specify the input samples", cli::type<std::string>());
    parser.add_option("o,output", "Specify the output file", cli::type<std::string>());
    parser.add_option("t,test", "Specify the test function", cli::type<std::string>());

    auto result = parser.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << parser.help() << '\n';
        return 0;
    }

    auto samples = result["input"].as<std::string>();
    auto log     = result["output"].as<std::string>();

    std::ifstream input(samples);
    std::ofstream output(log);

    if (input.is_open() == false)
    {
        std::cerr << samples << " is not a valid input\n";
        return -1;
    }

    auto func = result["test"].as<std::string>();
    auto type = result["algo"].as<std::string>();

    if (test.find(func) == test.end())
    {
        std::cerr << func << " is not a valid test function\n";
        return -1;
    }

    if (algo.find(type) == algo.end())
    {
        std::cerr << type << " is not a valid algorithm type\n";
        return -1;
    }

    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto creator = nw::creator(algo[type]);

        auto begin = std::chrono::high_resolution_clock::now();

        auto nw    = creator.create(match, miss, gap);
        int  score = std::invoke(test[func], *nw, ref, src);

        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        std::cout << "Exec Time: " << elapsed.count() << '\n';
        output << src.size() << ',' << score << ',' << elapsed.count() << '\n';
    }

    std::cout << "Testing has been completed!\n";
    return 0;
}
