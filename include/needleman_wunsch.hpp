#ifndef NEEDLEMAN_WUNSCH_HPP
#define NEEDLEMAN_WUNSCH_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <string>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

class NeedlemanWunsch
{
public:
    NeedlemanWunsch(int match, int miss, int gap);
    int score(std::string ref, std::string src);

private:
    int match;
    int miss;
    int gap;
};

#endif  // NEEDLEMAN_WUNSCH_HPP
