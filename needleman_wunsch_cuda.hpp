#ifndef NEEDLEMAN_WUNSCH_CUDA_HPP
#define NEEDLEMAN_WUNSCH_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "needleman_wunsch.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

class NeedlemanWunschCUDA : public NeedlemanWunsch
{
public:
    NeedlemanWunschCUDA(int match, int miss, int gap);
    int score(std::string ref, std::string src);
};

#endif  // NEEDLEMAN_WUNSCH_CUDA_HPP
