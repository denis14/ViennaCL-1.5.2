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
CMAKE_SOURCE_DIR = /home/denis/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/denis/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/depend.make

# Include the progress variables for this target.
include examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/flags.make

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/flags.make
examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o: ../examples/benchmarks/scheduler.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o -c /home/denis/ViennaCL-1.5.2/examples/benchmarks/scheduler.cpp

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/examples/benchmarks/scheduler.cpp > CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.i

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/examples/benchmarks/scheduler.cpp -o CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.s

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.requires:
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.requires

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.provides: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.requires
	$(MAKE) -f examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/build.make examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.provides.build
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.provides

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.provides.build: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o

# Object files for target schedulerbench-cpu
schedulerbench__cpu_OBJECTS = \
"CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o"

# External object files for target schedulerbench-cpu
schedulerbench__cpu_EXTERNAL_OBJECTS =

examples/benchmarks/schedulerbench-cpu: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o
examples/benchmarks/schedulerbench-cpu: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/build.make
examples/benchmarks/schedulerbench-cpu: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable schedulerbench-cpu"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/schedulerbench-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/build: examples/benchmarks/schedulerbench-cpu
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/build

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/requires: examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/scheduler.cpp.o.requires
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/requires

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/schedulerbench-cpu.dir/cmake_clean.cmake
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/clean

examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/benchmarks /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/benchmarks /home/denis/ViennaCL-1.5.2/build/examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/benchmarks/CMakeFiles/schedulerbench-cpu.dir/depend

