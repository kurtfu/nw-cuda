/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "cxxopts.hpp"
#include "nw/creator.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

class Profiler
{
public:
    using method = int (nw::aligner::*)(std::string const&, std::string const&);

    Profiler(std::string const& samples, std::string const& log)
        : input{samples}
        , output{log}
    {
        if (input.fail())
        {
            throw std::runtime_error("\'" + samples + "\' does not exist");
        }

        if (output.fail())
        {
            throw std::runtime_error("\'" + log + "\' could not be opened");
        }
    }

    void profile_samples(std::string const& type, std::string const& test)
    {
        validate_arguments(type, test);

        auto approach = approaches[type];
        auto func = methods[test];

        std::string line;

        while (std::getline(input, line))
        {
            std::istringstream iss(line);

            std::string src;
            std::string ref;

            iss >> src >> ref;

            auto begin = std::chrono::high_resolution_clock::now();

            auto nw = nw::creator(approach).create(1, -1, -2);
            int score = std::invoke(func, *nw, ref, src);

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

            std::cout << "Exec Time: " << elapsed.count() << '\n';
            output << src.size() << ',' << score << ',' << elapsed.count() << '\n';
        }
    }

private:
    void validate_arguments(std::string const& type, std::string const& test)
    {
        if (approaches.find(type) == approaches.end())
        {
            throw std::runtime_error("\'" + type + "\' is not a valid approach");
        }

        if (methods.find(test) == methods.end())
        {
            throw std::runtime_error("\'" + test + "\' is not a valid test");
        }
    }

    static std::unordered_map<std::string, Profiler::method> methods;
    static std::unordered_map<std::string, nw::approach> approaches;

    std::ifstream input;
    std::ofstream output;
};

std::unordered_map<std::string, Profiler::method> Profiler::methods = {
    {"fill",  &nw::aligner::fill },
    {"score", &nw::aligner::score}
};

std::unordered_map<std::string, nw::approach> Profiler::approaches = {
    {"cuda",   nw::approach::cuda  },
    {"serial", nw::approach::serial},
};

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

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    try
    {
        auto args = parse_program_argumnets(argc, argv);

        auto approach = args["approach"].as<std::string>();
        auto test = args["test"].as<std::string>();

        auto samples = args["input"].as<std::string>();
        auto log = args["output"].as<std::string>();

        Profiler profiler(samples, log);

        profiler.profile_samples(approach, test);
        std::cout << "Testing has been completed!\n";
    }
    catch (std::exception const& ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return 0;
}
