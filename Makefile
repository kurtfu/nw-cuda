#------------------------------------------------------------------------------
# PROJECT CONFIGURATIONS
#------------------------------------------------------------------------------

# The tag describes the name of the project.
PROJ = $(notdir ${CURDIR})

#------------------------------------------------------------------------------
# PATH DEFINITIONS
#------------------------------------------------------------------------------

PROJ_PATH  = .
BIN_PATH   = $(addsuffix /bin, ${PROJ_PATH})
BUILD_PATH = $(addsuffix /build, ${PROJ_PATH})

#------------------------------------------------------------------------------
# SHELL COMMANDS
#------------------------------------------------------------------------------

ifeq (${OS}, Windows_NT)
    MKDIR = mkdir 2>nul     # Make directory and suspend any error.
    RMDIR = rd /s /q 2>nul  # Remove directory and suspend any error.

    OBJ_EXT = .obj          # Object file suffix.
else
    MKDIR = mkdir -p        # Make directory and suspend any error.
    RMDIR = rm -rf          # Remove directory and suspend any error.

    OBJ_EXT = .o            # Object file suffix.
endif

#------------------------------------------------------------------------------
# INPUT & OUTPUT FILE DEFINITIONS
#------------------------------------------------------------------------------

# The tag describes the search path for header files of the project.
IPATH = ${PROJ_PATH}/include

# The tag describes the source files of the project.
SRC  = $(wildcard ${PROJ_PATH}/*.cpp)     \
       $(wildcard ${PROJ_PATH}/src/*.cpp) \
       $(wildcard ${PROJ_PATH}/src/*.cu)

# The tag describes the object files of the project.
OBJ  = $(patsubst ${PROJ_PATH}/%.cpp,${BUILD_PATH}/%${OBJ_EXT}, ${SRC})
OBJ := $(patsubst ${PROJ_PATH}/%.cu,${BUILD_PATH}/%${OBJ_EXT}, ${OBJ})

# The tag describes the output file of the project.
OUT  = $(addprefix ${BIN_PATH}/, ${PROJ})

#------------------------------------------------------------------------------
# BUILD TOOLS
#------------------------------------------------------------------------------

CC = nvcc  # CUDA/C++ Compiler
LD = nvcc  # Linker

#------------------------------------------------------------------------------
# COMPILER & LINKER FLAGS
#------------------------------------------------------------------------------

CXXFLAGS = $(addprefix -I, ${IPATH}) \
           -expt-relaxed-constexpr \
           -std=c++17 \
           -O2

#------------------------------------------------------------------------------
# MAKE RULES
#------------------------------------------------------------------------------

.PHONY: all clean

all: ${OUT}
	@echo "Project Build Successfully"

clean:
	@${RMDIR} "${BIN_PATH}" ||:
	@${RMDIR} "${BUILD_PATH}" ||:
	@echo "Project Cleaned Successfully"

#------------------------------------------------------------------------------
# BUILD RULES
#------------------------------------------------------------------------------

${OUT}: ${OBJ}
	@${MKDIR} "$(dir $@)" ||:
	${LD} -o $@ ${OBJ} ${LD_FLAGS}

${BUILD_PATH}/%${OBJ_EXT}: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${CC} -c $< -o $@ ${CXXFLAGS}

${BUILD_PATH}/%${OBJ_EXT}: ${PROJ_PATH}/%.cu
	@${MKDIR} "$(dir $@)" ||:
	${CC} -c $< -o $@ ${CXXFLAGS}
