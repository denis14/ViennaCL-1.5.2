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
include examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/depend.make

# Include the progress variables for this target.
include examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/flags.make

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/flags.make
examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o: ../examples/benchmarks/blas3.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o -c /home/denis/ViennaCL-1.5.2/examples/benchmarks/blas3.cpp

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blas3bench-cpu.dir/blas3.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/examples/benchmarks/blas3.cpp > CMakeFiles/blas3bench-cpu.dir/blas3.cpp.i

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blas3bench-cpu.dir/blas3.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/examples/benchmarks/blas3.cpp -o CMakeFiles/blas3bench-cpu.dir/blas3.cpp.s

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.requires:
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.requires

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.provides: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.requires
	$(MAKE) -f examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/build.make examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.provides.build
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.provides

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.provides.build: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o

# Object files for target blas3bench-cpu
blas3bench__cpu_OBJECTS = \
"CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o"

# External object files for target blas3bench-cpu
blas3bench__cpu_EXTERNAL_OBJECTS =

examples/benchmarks/blas3bench-cpu: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o
examples/benchmarks/blas3bench-cpu: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/build.make
examples/benchmarks/blas3bench-cpu: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable blas3bench-cpu"
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blas3bench-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/build: examples/benchmarks/blas3bench-cpu
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/build

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/requires: examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/blas3.cpp.o.requires
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/requires

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/blas3bench-cpu.dir/cmake_clean.cmake
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/clean

examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/benchmarks /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/benchmarks /home/denis/ViennaCL-1.5.2/build/examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/benchmarks/CMakeFiles/blas3bench-cpu.dir/depend

