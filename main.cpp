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
/*  DATA TYPES                                                               */
/*****************************************************************************/

class Profiler
{
    using method = int (nw::aligner::*)(std::string const&, std::string const&);

    struct Result
    {
        int score;
        std::chrono::milliseconds duration;
    };

public:
    Profiler(std::string const& approach, std::string const& test)
    {
        add_creator(approach);
        add_test(test);
    }

    void add_input(std::string const& file)
    {
        input.open(file);

        if (input.fail())
        {
            throw std::runtime_error("\'" + file + "\' does not exist");
        }
    }

    void add_output(std::string const& file)
    {
        output.open(file);

        if (output.fail())
        {
            throw std::runtime_error("\'" + file + "\' could not be opened");
        }
    }

    void profile_samples()
    {
        std::string line;

        while (std::getline(input, line))
        {
            std::istringstream iss(line);

            std::string src;
            std::string ref;

            iss >> src >> ref;

            auto result = measure_sample(ref, src);

            auto score    = result.score;
            auto duration = result.duration.count();

            std::cout << "Exec Time: " << duration << '\n';
            output << src.size() << ',' << score << ',' << duration << '\n';
        }
    }

private:
    void add_creator(std::string const& type)
    {
        static std::unordered_map<std::string, nw::approach> approaches = {
            {"cuda",   nw::approach::cuda  },
            {"serial", nw::approach::serial},
        };

        if (approaches.find(type) == approaches.end())
        {
            throw std::runtime_error("\'" + type + "\' is not a valid");
        }

        creator = std::make_unique<nw::creator>(approaches[type]);
    }

    void add_test(std::string const& type)
    {
        static std::unordered_map<std::string, Profiler::method> tests = {
            {"fill",  &nw::aligner::fill },
            {"score", &nw::aligner::score}
        };

        if (tests.find(type) == tests.end())
        {
            throw std::runtime_error("\'" + type + "\' is not a valid");
        }

        test = tests[type];
    }

    Result measure_sample(std::string const& ref, std::string const& src)
    {
        constexpr int match = 1;
        constexpr int miss  = -1;
        constexpr int gap   = -2;

        auto begin = std::chrono::high_resolution_clock::now();

        auto nw    = creator->create(match, miss, gap);
        int  score = std::invoke(test, *nw, ref, src);

        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        return Result{score, elapsed};
    }

    std::unique_ptr<nw::creator> creator;
    Profiler::method             test;

    std::ifstream input;
    std::ofstream output;
};

/*****************************************************************************/
/*  MODULE FUNCTIONS                                                         */
/*****************************************************************************/

cli::result parse_program_argumnets(int argc, char const* argv[])
{
    cli::parser parser("nw");

    auto type = cli::type<std::string>();

    parser.add_option("a,approach", "Specify the approach", type, "APPROACH");
    parser.add_option("h,help", "Display help");
    parser.add_option("i,input", "Specify the input samples", type, "FILE");
    parser.add_option("o,output", "Specify the output file", type, "FILE");
    parser.add_option("t,test", "Specify the test method", type, "METHOD");

    auto args = parser.parse(argc, argv);

    if (args.count("help"))
    {
        std::cout << parser.help() << '\n';
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
        auto test     = args["test"].as<std::string>();

        Profiler profiler(approach, test);

        auto samples = args["input"].as<std::string>();
        auto log     = args["output"].as<std::string>();

        profiler.add_input(samples);
        profiler.add_output(log);

        profiler.profile_samples();
        std::cout << "Testing has been completed!\n";
    }
    catch (std::exception const& ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return 0;
}
