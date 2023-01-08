/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "profiler/base.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using profiler::base;

/*****************************************************************************/
/*  TYPE ALIASES                                                             */
/*****************************************************************************/

using scale = std::chrono::milliseconds;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

void base::assign_scoring_coefficients(int match, int miss, int gap)
{
    this->match = match;
    this->miss = miss;
    this->gap = gap;
}

void base::assign_nw_approach(nw::approach type)
{
    nw_approach = type;
}

void base::attach_input(std::string const& file)
{
    input.open(file);

    if (input.fail())
    {
        throw std::runtime_error("\'" + file + "\' does not exist");
    }
}

void base::attach_output(std::string const& file)
{
    output.open(file);

    if (output.fail())
    {
        throw std::runtime_error("\'" + file + "\' could not be opened");
    }
}

void base::profile_samples()
{
    std::string line;

    while (std::getline(input, line))
    {
        std::istringstream iss(line);

        std::string src;
        std::string ref;

        iss >> src >> ref;

        auto begin = std::chrono::high_resolution_clock::now();

        auto result = run(nw::input{ref}, nw::input{src});

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<scale>(end - begin);

        std::cout << "Exec Time: " << time.count() << '\n';
        output << src.length() << ',' << time.count() << ',' << result << '\n';
    }
}
