# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/denis/Documents/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/denis/Documents/ViennaCL-1.5.2/tests

# Include any dependencies generated for this target.
include examples/autotuner/CMakeFiles/gemv_autotuning.dir/depend.make

# Include the progress variables for this target.
include examples/autotuner/CMakeFiles/gemv_autotuning.dir/progress.make

# Include the compile flags for this target's objects.
include examples/autotuner/CMakeFiles/gemv_autotuning.dir/flags.make

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o: examples/autotuner/CMakeFiles/gemv_autotuning.dir/flags.make
examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o: ../examples/autotuner/gemv_autotuning.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/Documents/ViennaCL-1.5.2/tests/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o"
	cd /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o -c /home/denis/Documents/ViennaCL-1.5.2/examples/autotuner/gemv_autotuning.cpp

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.i"
	cd /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/Documents/ViennaCL-1.5.2/examples/autotuner/gemv_autotuning.cpp > CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.i

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.s"
	cd /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/Documents/ViennaCL-1.5.2/examples/autotuner/gemv_autotuning.cpp -o CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.s

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.requires:
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.requires

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.provides: examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.requires
	$(MAKE) -f examples/autotuner/CMakeFiles/gemv_autotuning.dir/build.make examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.provides.build
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.provides

examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.provides.build: examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o

# Object files for target gemv_autotuning
gemv_autotuning_OBJECTS = \
"CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o"

# External object files for target gemv_autotuning
gemv_autotuning_EXTERNAL_OBJECTS =

examples/autotuner/gemv_autotuning: examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o
examples/autotuner/gemv_autotuning: examples/autotuner/CMakeFiles/gemv_autotuning.dir/build.make
examples/autotuner/gemv_autotuning: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/autotuner/gemv_autotuning: examples/autotuner/CMakeFiles/gemv_autotuning.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable gemv_autotuning"
	cd /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemv_autotuning.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/autotuner/CMakeFiles/gemv_autotuning.dir/build: examples/autotuner/gemv_autotuning
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/build

examples/autotuner/CMakeFiles/gemv_autotuning.dir/requires: examples/autotuner/CMakeFiles/gemv_autotuning.dir/gemv_autotuning.cpp.o.requires
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/requires

examples/autotuner/CMakeFiles/gemv_autotuning.dir/clean:
	cd /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner && $(CMAKE_COMMAND) -P CMakeFiles/gemv_autotuning.dir/cmake_clean.cmake
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/clean

examples/autotuner/CMakeFiles/gemv_autotuning.dir/depend:
	cd /home/denis/Documents/ViennaCL-1.5.2/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/Documents/ViennaCL-1.5.2 /home/denis/Documents/ViennaCL-1.5.2/examples/autotuner /home/denis/Documents/ViennaCL-1.5.2/tests /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner /home/denis/Documents/ViennaCL-1.5.2/tests/examples/autotuner/CMakeFiles/gemv_autotuning.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/autotuner/CMakeFiles/gemv_autotuning.dir/depend

