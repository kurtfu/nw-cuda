#------------------------------------------------------------------------------
# PROJECT CONFIGURATIONS
#------------------------------------------------------------------------------

# The tag describes the name of the project.
PROJ = seqgen

#------------------------------------------------------------------------------
# SHELL COMMANDS
#------------------------------------------------------------------------------

ifeq (${OS}, Windows_NT)
    MKDIR = mkdir 2>nul     # Make directory and suspend any error.
    RMDIR = rd /s /q 2>nul  # Remove directory and suspend any error.
else
    MKDIR = mkdir -p        # Make directory and suspend any error.
    RMDIR = rm -rf          # Remove directory and suspend any error.
endif

#------------------------------------------------------------------------------
# INPUT & OUTPUT FILE DEFINITIONS
#------------------------------------------------------------------------------

# The tag describes the search path for header files of the project.
IPATH = ${PROJ_PATH}/include

# The tag describes the source files of the project.
SRC  = $(wildcard ${PROJ_PATH}/utils/seqgen.cpp)

# The tag describes the object files of the project.
OBJ  = $(patsubst ${PROJ_PATH}/%.cpp,${BUILD_PATH}/%.o, ${SRC})

# The tag describes the output file of the project.
OUT  = $(addprefix ${BIN_PATH}/, ${PROJ})

#------------------------------------------------------------------------------
# BUILD TOOLS
#------------------------------------------------------------------------------

CC = g++  # C++ Compiler
LD = g++  # Linker

#------------------------------------------------------------------------------
# COMPILER & LINKER FLAGS
#------------------------------------------------------------------------------

CXXFLAGS = $(addprefix -I, ${IPATH}) \
           -Wall \
           -Wextra \
           -std=c++17 \
           -O2

#------------------------------------------------------------------------------
# MAKE RULES
#------------------------------------------------------------------------------

.PHONY: seqgen

seqgen: ${OUT}
	@echo "Sequence Generator Build Successfully"

#------------------------------------------------------------------------------
# BUILD RULES
#------------------------------------------------------------------------------

${OUT}: ${OBJ}
	@${MKDIR} "$(dir $@)" ||:
	${LD} -o $@ ${OBJ} ${LD_FLAGS}

${BUILD_PATH}/%.o: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${CC} -c $< -o $@ ${CXXFLAGS}
