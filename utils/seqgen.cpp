/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

/*****************************************************************************/
/*  MAIN APPLICATION                                                         */
/*****************************************************************************/

int main(int argc, char const* argv[])
{
    auto find_opt = [&](std::string target)
    {
        auto opt = std::find(argv, argv + argc, target);
        return (opt == argv + argc || (opt + 1) == argv + argc) ? nullptr : opt;
    };

    auto opt = find_opt("--length");
    std::size_t length = (opt == nullptr) ? 1000 : std::atoi(*(opt + 1));

    opt = find_opt("--sample");
    std::size_t sample_count = (opt == nullptr) ? 10 : std::atoi(*(opt + 1));

    auto random_sequence = [](std::size_t length)
    {
        constexpr std::string_view amino_acid = "acdefghiklmnpqrstwyv";

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<> distribution(0, amino_acid.size() - 1);

        std::string peptide;

        for (std::size_t i = 0; i < length; ++i)
        {
            peptide += amino_acid[distribution(generator)];
        }

        return peptide;
    };

    auto log = find_opt("--output");

    if (log == nullptr)
    {
        std::cerr << "Output file must be specified with \'--output\'\n";
        return - 1;
    }

    std::ofstream output(*(log + 1));

    for (std::size_t sample = 0; sample < sample_count; ++sample)
    {
        std::string src = random_sequence(length);
        std::string ref = random_sequence(length);

        output << src << ' ' << ref << '\n';
    }

    std::cout << "Sequences have been generated!\n";
    return 0;
}
