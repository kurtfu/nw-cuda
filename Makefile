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
else
    MKDIR = mkdir -p        # Make directory and suspend any error.
    RMDIR = rm -rf          # Remove directory and suspend any error.
endif

#------------------------------------------------------------------------------
# INPUT & OUTPUT FILE DEFINITIONS
#------------------------------------------------------------------------------

# The tag describes the search path for header files of the project.
IPATH = ${PROJ_PATH}/include \
        ${PROJ_PATH}/vendor/cxxopts/include

# The tag describes the source files of the project.
SRC  = $(wildcard ${PROJ_PATH}/*.cpp)         \
       $(wildcard ${PROJ_PATH}/src/nw/*.cpp)  \
       $(wildcard ${PROJ_PATH}/src/nw/*.cu)

# The tag describes the object files of the project.
OBJ  = $(patsubst ${PROJ_PATH}/%.cpp,${BUILD_PATH}/%.o, ${SRC})
OBJ := $(patsubst ${PROJ_PATH}/%.cu,${BUILD_PATH}/%.o, ${OBJ})

# The tag describes the output file of the project.
OUT  = $(addprefix ${BIN_PATH}/, ${PROJ})

# Convert object suffix to MSVC style if the host is Windows.
ifeq (${OS}, Windows_NT)
    OBJ := $(OBJ:.o=.obj)
endif

#------------------------------------------------------------------------------
# BUILD TOOLS
#------------------------------------------------------------------------------

NVCC = nvcc  # CUDA/C++ Compiler

#------------------------------------------------------------------------------
# COMPILER & LINKER FLAGS
#------------------------------------------------------------------------------

NVCCFLAGS = $(addprefix -I, ${IPATH}) \
            -expt-relaxed-constexpr \
            -std=c++17 \
            -O2

#------------------------------------------------------------------------------
# MAKE RULES
#------------------------------------------------------------------------------

.PHONY: all clean seqgen

all: ${OUT}
	@echo "Project Build Successfully"

clean:
	@${RMDIR} "${BIN_PATH}" ||:
	@${RMDIR} "${BUILD_PATH}" ||:
	@echo "Project Cleaned Successfully"

seqgen:
	@$(MAKE) --no-print-directory -C utils

#------------------------------------------------------------------------------
# BUILD RULES
#------------------------------------------------------------------------------

${OUT}: ${OBJ}
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -o $@ ${OBJ}

${BUILD_PATH}/%.o: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -c $< -o $@ ${NVCCFLAGS}

${BUILD_PATH}/%.o: ${PROJ_PATH}/%.cu
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -c $< -o $@ ${NVCCFLAGS}

${BUILD_PATH}/%.obj: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -c $< -o $@ ${NVCCFLAGS}

${BUILD_PATH}/%.obj: ${PROJ_PATH}/%.cu
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -c $< -o $@ ${NVCCFLAGS}
