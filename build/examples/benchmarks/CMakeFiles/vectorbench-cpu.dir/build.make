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
CMAKE_SOURCE_DIR = /home/andi/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/depend.make

# Include the progress variables for this target.
include examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/flags.make

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/flags.make
examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o: ../examples/benchmarks/vector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o"
	cd /home/andi/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vectorbench-cpu.dir/vector.cpp.o -c /home/andi/ViennaCL-1.5.2/examples/benchmarks/vector.cpp

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vectorbench-cpu.dir/vector.cpp.i"
	cd /home/andi/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/ViennaCL-1.5.2/examples/benchmarks/vector.cpp > CMakeFiles/vectorbench-cpu.dir/vector.cpp.i

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vectorbench-cpu.dir/vector.cpp.s"
	cd /home/andi/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/ViennaCL-1.5.2/examples/benchmarks/vector.cpp -o CMakeFiles/vectorbench-cpu.dir/vector.cpp.s

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.requires:
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.requires

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.provides: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.requires
	$(MAKE) -f examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/build.make examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.provides.build
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.provides

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.provides.build: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o

# Object files for target vectorbench-cpu
vectorbench__cpu_OBJECTS = \
"CMakeFiles/vectorbench-cpu.dir/vector.cpp.o"

# External object files for target vectorbench-cpu
vectorbench__cpu_EXTERNAL_OBJECTS =

examples/benchmarks/vectorbench-cpu: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o
examples/benchmarks/vectorbench-cpu: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/build.make
examples/benchmarks/vectorbench-cpu: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable vectorbench-cpu"
	cd /home/andi/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vectorbench-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/build: examples/benchmarks/vectorbench-cpu
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/build

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/requires: examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/vector.cpp.o.requires
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/requires

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/clean:
	cd /home/andi/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/vectorbench-cpu.dir/cmake_clean.cmake
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/clean

examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/depend:
	cd /home/andi/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/ViennaCL-1.5.2 /home/andi/ViennaCL-1.5.2/examples/benchmarks /home/andi/ViennaCL-1.5.2/build /home/andi/ViennaCL-1.5.2/build/examples/benchmarks /home/andi/ViennaCL-1.5.2/build/examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/benchmarks/CMakeFiles/vectorbench-cpu.dir/depend

