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
include examples/tutorial/CMakeFiles/bandwidth-reduction.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/bandwidth-reduction.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/bandwidth-reduction.dir/flags.make

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/flags.make
examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o: ../examples/tutorial/bandwidth-reduction.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o -c /home/denis/ViennaCL-1.5.2/examples/tutorial/bandwidth-reduction.cpp

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/examples/tutorial/bandwidth-reduction.cpp > CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.i

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/examples/tutorial/bandwidth-reduction.cpp -o CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.s

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.requires:
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.requires

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.provides: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.requires
	$(MAKE) -f examples/tutorial/CMakeFiles/bandwidth-reduction.dir/build.make examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.provides.build
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.provides

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.provides.build: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o

# Object files for target bandwidth-reduction
bandwidth__reduction_OBJECTS = \
"CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o"

# External object files for target bandwidth-reduction
bandwidth__reduction_EXTERNAL_OBJECTS =

examples/tutorial/bandwidth-reduction: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o
examples/tutorial/bandwidth-reduction: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/build.make
examples/tutorial/bandwidth-reduction: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/tutorial/bandwidth-reduction: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bandwidth-reduction"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bandwidth-reduction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/bandwidth-reduction.dir/build: examples/tutorial/bandwidth-reduction
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/build

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/requires: examples/tutorial/CMakeFiles/bandwidth-reduction.dir/bandwidth-reduction.cpp.o.requires
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/requires

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/bandwidth-reduction.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/clean

examples/tutorial/CMakeFiles/bandwidth-reduction.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/tutorial /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/tutorial /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/bandwidth-reduction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/bandwidth-reduction.dir/depend

