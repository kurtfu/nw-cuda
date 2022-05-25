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
       $(wildcard ${PROJ_PATH}/src/nw/*.cu)   \
       $(wildcard ${PROJ_PATH}/test/catch2/*.cpp) \
       $(wildcard ${PROJ_PATH}/test/*.cpp)

# The tag describes the object files of the project.
OBJ  = $(patsubst ${PROJ_PATH}/%.cpp,${BUILD_PATH}/%, ${SRC})
OBJ := $(patsubst ${PROJ_PATH}/%.cu,${BUILD_PATH}/%, ${OBJ})

# The tag describes the dependency files of the sources.
DEP  = $(patsubst ${PROJ_PATH}/%.cpp,${BUILD_PATH}/%.d, ${SRC})
DEP := $(patsubst ${PROJ_PATH}/%.cu,${BUILD_PATH}/%.d, ${DEP})

# The tag describes the output file of the project.
OUT  = $(addprefix ${BIN_PATH}/, ${PROJ})

# The tag describes the test executable of the project.
TEST = $(addprefix ${BIN_PATH}/, ${PROJ}-test)

#------------------------------------------------------------------------------
# EXTENSION ALIGNMENTS
#------------------------------------------------------------------------------

ifeq (${OS}, Windows_NT)
    OBJ  := $(addsuffix .obj,${OBJ})
    OUT  := $(addsuffix .exe,${OUT})
    TEST := $(addsuffix .exe,${TEST})
else
    OBJ  := $(addsuffix .o,${OBJ})
    OUT  := $(addsuffix .out,${OUT})
    TEST := $(addsuffix .out,${TEST})
endif

#------------------------------------------------------------------------------
# BUILD TOOLS
#------------------------------------------------------------------------------

NVCC = nvcc  # CUDA/C++ Compiler

#------------------------------------------------------------------------------
# COMPILER & LINKER FLAGS
#------------------------------------------------------------------------------

NVCCFLAGS = $(addprefix -I, ${IPATH}) \
            -std=c++17 \
            -MD \
            -O2

#------------------------------------------------------------------------------
# MAKE RULES
#------------------------------------------------------------------------------

all: OBJ := $(filter-out ${BUILD_PATH}/test/%.obj,${OBJ})
all: OBJ := $(filter-out ${BUILD_PATH}/test/%.o,${OBJ})
all: ${OUT}
	@echo "Project Build Successfully"
.PHONY: all

clean:
	@${RMDIR} "${BIN_PATH}" ||:
	@${RMDIR} "${BUILD_PATH}" ||:
	@echo "Project Cleaned Successfully"
.PHONY: clean

test: OBJ := $(filter-out ${BUILD_PATH}/main.obj,${OBJ})
test: OBJ := $(filter-out ${BUILD_PATH}/main.o,${OBJ})
test: ${TEST}
	@"${TEST}" ||:
.PHONY: test

#------------------------------------------------------------------------------
# BUILD RULES
#------------------------------------------------------------------------------

${OUT}: ${OBJ}
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -o $@ ${OBJ}

${TEST}: ${OBJ}
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} -o $@ ${OBJ}

${BUILD_PATH}/%.o: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${BUILD_PATH}/%.o: ${PROJ_PATH}/%.cu
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${BUILD_PATH}/%.obj: ${PROJ_PATH}/%.cpp
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${BUILD_PATH}/%.obj: ${PROJ_PATH}/%.cu
	@${MKDIR} "$(dir $@)" ||:
	${NVCC} ${NVCCFLAGS} -c $< -o $@

-include ${DEP}
