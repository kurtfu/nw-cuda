/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "creator.hpp"

/*****************************************************************************/
/*  TYPE ALIASES                                                             */
/*****************************************************************************/

using test_signature = void(nw::algo, std::ifstream&, std::ofstream&);

/*****************************************************************************/
/*  MODULE VARIABLES                                                         */
/*****************************************************************************/

constexpr int match = 1;
constexpr int miss  = -1;
constexpr int gap   = -2;

/*****************************************************************************/
/*  TEST FUNCTIONS                                                           */
/*****************************************************************************/

void fill(nw::algo algo, std::ifstream& input, std::ofstream& output)
{
    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto creator = nw::creator(algo);

        auto begin = std::chrono::high_resolution_clock::now();

        auto nw    = creator.create(match, miss, gap);
        int  score = nw->fill(ref, src);

        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        std::size_t rw = nw->row_count() - 1;
        std::size_t cl = nw->col_count() - 1;

        std::cout << "Exec Time: " << elapsed.count() << '\n';
        output << src.size() << ',' << score << ',' << elapsed.count() << '\n';
    }
}

void score(nw::algo algo, std::ifstream& input, std::ofstream& output)
{
    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto creator = nw::creator(algo);

        auto begin = std::chrono::high_resolution_clock::now();

        auto nw    = creator.create(match, miss, gap);
        int  score = nw->score(ref, src);

        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        std::size_t rw = nw->row_count() - 1;
        std::size_t cl = nw->col_count() - 1;

        std::cout << "Exec Time: " << elapsed.count() << '\n';
        output << src.size() << ',' << score << ',' << elapsed.count() << '\n';
    }
}

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    std::unordered_map<std::string, nw::algo> algo = {
        {"cuda",   nw::algo::cuda  },
        {"serial", nw::algo::serial},
    };

    std::unordered_map<std::string, std::function<test_signature>> test = {
        {"fill",  fill },
        {"score", score}
    };

    auto find_opt = [&](std::string&& target)
    {
        auto opt = std::find(argv, argv + argc, target);
        return (opt == argv + argc || (opt + 1) == argv + argc) ? nullptr : opt;
    };

    auto samples = find_opt("--input");

    if (samples == nullptr)
    {
        std::cerr << "Input file must be specified with \'--input\'\n";
        return -1;
    }

    auto log = find_opt("--output");

    if (log == nullptr)
    {
        std::cerr << "Output file must be specified with \'--output\'\n";
        return -1;
    }

    auto type = find_opt("--algo");

    if (type == nullptr)
    {
        std::cerr << "Algorithm type must be specified with \'--algo\'\n";
        return -1;
    }

    if (algo.find(*(type + 1)) == algo.end())
    {
        std::cerr << *(type + 1) << " is not a valid algorithm type\n";
        return -1;
    }

    std::ifstream input(*(samples + 1));
    std::ofstream output(*(log + 1));

    if (input.is_open() == false)
    {
        std::cerr << *(samples + 1) << " is not a valid input\n";
        return -1;
    }

    auto func = find_opt("--test");

    if (func == nullptr)
    {
        std::cerr << "Test function must be specified with \'--test\'\n";
        return -1;
    }

    if (test.find(*(func + 1)) == test.end())
    {
        std::cerr << *(func + 1) << " is not a valid test function\n";
        return -1;
    }

    test[*(func + 1)](algo[*(type + 1)], input, output);

    std::cout << "Testing has been completed!\n";
    return 0;
}
