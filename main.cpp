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
    using method = int (nw::aligner::*)(nw::input const&, nw::input const&);
    using scale = std::chrono::milliseconds;

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

    void profile_samples(nw::approach approach, std::string const& func)
    {
        if (func == "align")
        {
            execute(approach, &nw::aligner::align);
        }
        else if (func == "score")
        {
            execute(approach, &nw::aligner::score);
        }
    }

private:
    std::pair<nw::input, nw::input> parse_input_line(std::string const& line)
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        return std::make_pair(nw::input(ref), nw::input(src));
    }

    template <typename T>
    void execute(nw::approach approach, T func)
    {
        std::string line;

        while (std::getline(input, line))
        {
            auto sequences = parse_input_line(line);

            auto ref = sequences.first;
            auto src = sequences.second;

            auto begin = std::chrono::high_resolution_clock::now();

            auto nw = nw::creator(approach).create(1, -1, -2);
            auto score = std::invoke(func, *nw, ref, src);

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<scale>(end - begin);

            auto exec_time = elapsed.count();

            std::cout << "Exec Time: " << exec_time << '\n';
            output << src.length() << ',' << score << ',' << exec_time << '\n';
        }
    }

    std::ifstream input;
    std::ofstream output;
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

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    try
    {
        auto args = parse_program_argumnets(argc, argv);

        auto approach = args["approach"].as<nw::approach>();
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
