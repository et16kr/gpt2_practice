APP_NAME ?= inference_practice
SRC_DIR ?= src
BUILD_DIR ?= out

CXX ?= g++
CUDACXX ?= nvcc
CPPFLAGS ?= -I$(SRC_DIR)
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -MMD -MP
NVCCFLAGS ?= -std=c++17 -O2 -MMD -MP
LDFLAGS ?=
LDLIBS ?= -lcudart

CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS)) \
	$(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
DEPS := $(OBJS:.o=.d)
TARGET := $(BUILD_DIR)/$(APP_NAME)
LINKER := $(if $(strip $(CU_SRCS)),$(CUDACXX),$(CXX))

.PHONY: all build run clean help

all: build

help:
	@echo "Targets:"
	@echo "  make / make build  - Build $(TARGET)"
	@echo "  make run           - Build and run $(TARGET)"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "Variables (override): APP_NAME, CXX, CUDACXX, CPPFLAGS, CXXFLAGS, NVCCFLAGS, LDFLAGS, LDLIBS"

build: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(LINKER) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(CUDACXX) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
