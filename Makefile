#------------------------------------------------------------------------------
# PROJECT CONFIGURATIONS
#------------------------------------------------------------------------------

# The name of the project
PROJ := $(notdir $(CURDIR))

#------------------------------------------------------------------------------
# PATH DEFINITIONS
#------------------------------------------------------------------------------

PROJ_PATH = .

BIN_DIR   = $(PROJ_PATH)/bin
BUILD_DIR = $(PROJ_PATH)/build

#------------------------------------------------------------------------------
# BUILD TOOLS
#------------------------------------------------------------------------------

NVCC = nvcc  # CUDA/C++ Compiler

#------------------------------------------------------------------------------
# COMPILER & LINKER FLAGS
#------------------------------------------------------------------------------

NVCCFLAGS = $(addprefix -I, $(IPATH)) \
            -std=c++17 \
            -MD \
            -O2

#------------------------------------------------------------------------------
# INPUT FILE DEFINITIONS
#------------------------------------------------------------------------------

# Search path for header files of the project
IPATH = $(PROJ_PATH)/include \
        $(PROJ_PATH)/vendor/cxxopts/include

# Source list of the project
SRC = $(wildcard $(PROJ_PATH)/src/nw/*.cpp)  \
      $(wildcard $(PROJ_PATH)/src/nw/*.cu)   \
      $(wildcard $(PROJ_PATH)/src/profiler/*.cpp)

ifneq ($(MAKECMDGOALS), test)
    SRC += $(PROJ_PATH)/main.cpp
else
    SRC += $(wildcard $(PROJ_PATH)/test/*.cpp) \
           $(wildcard $(PROJ_PATH)/test/catch2/*.cpp)
endif

#------------------------------------------------------------------------------
# OUTPUT FILE DEFINITIONS
#------------------------------------------------------------------------------

# Object list of the project
OBJ  = $(patsubst $(PROJ_PATH)/%.cpp,$(BUILD_DIR)/%, $(SRC))
OBJ := $(patsubst $(PROJ_PATH)/%.cu,$(BUILD_DIR)/%, $(OBJ))

# Dependency list of the modules
DEP  = $(patsubst $(PROJ_PATH)/%.cpp,$(BUILD_DIR)/%.d, $(SRC))
DEP := $(patsubst $(PROJ_PATH)/%.cu,$(BUILD_DIR)/%.d, $(DEP))

# The executable output of the project
OUT  = $(addprefix $(BIN_DIR)/, $(PROJ))

# Unit tests of the project
TEST = $(addprefix $(BIN_DIR)/, $(PROJ)-test)

#------------------------------------------------------------------------------
# EXTENSION ALIGNMENTS
#------------------------------------------------------------------------------

ifeq ($(OS), Windows_NT)
    OBJ  := $(addsuffix .obj,$(OBJ))
    OUT  := $(addsuffix .exe,$(OUT))
    TEST := $(addsuffix .exe,$(TEST))
else
    OBJ  := $(addsuffix .o,$(OBJ))
    OUT  := $(addsuffix .out,$(OUT))
    TEST := $(addsuffix .out,$(TEST))
endif

#------------------------------------------------------------------------------
# SHELL COMMANDS
#------------------------------------------------------------------------------

ifeq ($(OS), Windows_NT)
    SHELL = cmd
    MKDIR = mkdir 2>nul
    RMDIR = rd /s /q 2>nul
else
    SHELL = /bin/sh
    MKDIR = mkdir -p
    RMDIR = rm -rf
endif

#------------------------------------------------------------------------------
# MAKE RULES
#------------------------------------------------------------------------------

all: $(OUT)
	@echo "  Built target $<"
.PHONY: all

clean:
	@$(RMDIR) "$(BIN_DIR)" ||:
	@$(RMDIR) "$(BUILD_DIR)" ||:
	@echo "  Clean finished"
.PHONY: clean

test: $(TEST)
	@echo "  Testing..."
	@"$(TEST)" ||:
.PHONY: test

#------------------------------------------------------------------------------
# BUILD RULES
#------------------------------------------------------------------------------

$(OUT): $(OBJ)
	@echo "  Linking CUDA executable $@"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -o $@ $(OBJ)

$(TEST): $(OBJ)
	@echo "  Linking unit tests"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -o $@ $(OBJ)

$(BUILD_DIR)/%.o: $(PROJ_PATH)/%.cpp
	@echo "  Building CXX object $@"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -c $< -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/%.o: $(PROJ_PATH)/%.cu
	@echo "  Building CUDA object $@"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -c $< -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/%.obj: $(PROJ_PATH)/%.cpp
	@echo "  Building CXX object $@"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -c $< -o $@ $(NVCCFLAGS)

$(BUILD_DIR)/%.obj: $(PROJ_PATH)/%.cu
	@echo "  Building CUDA object $@"
	@$(MKDIR) "$(dir $@)" ||:
	@$(NVCC) -c $< -o $@ $(NVCCFLAGS)

-include $(DEP)
