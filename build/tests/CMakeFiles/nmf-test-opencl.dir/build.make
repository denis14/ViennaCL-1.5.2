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
CMAKE_SOURCE_DIR = /home/andi/git/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/nmf-test-opencl.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/nmf-test-opencl.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/nmf-test-opencl.dir/flags.make

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o: tests/CMakeFiles/nmf-test-opencl.dir/flags.make
tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o: ../tests/src/nmf.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o -c /home/andi/git/ViennaCL-1.5.2/tests/src/nmf.cpp

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.i"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/ViennaCL-1.5.2/tests/src/nmf.cpp > CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.i

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.s"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/ViennaCL-1.5.2/tests/src/nmf.cpp -o CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.s

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.requires:
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.requires

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.provides: tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/nmf-test-opencl.dir/build.make tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.provides.build
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.provides

tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.provides.build: tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o

# Object files for target nmf-test-opencl
nmf__test__opencl_OBJECTS = \
"CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o"

# External object files for target nmf-test-opencl
nmf__test__opencl_EXTERNAL_OBJECTS =

tests/nmf-test-opencl: tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o
tests/nmf-test-opencl: tests/CMakeFiles/nmf-test-opencl.dir/build.make
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libOpenCL.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libboost_system.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/nmf-test-opencl: /usr/lib/i386-linux-gnu/libpthread.so
tests/nmf-test-opencl: tests/CMakeFiles/nmf-test-opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable nmf-test-opencl"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nmf-test-opencl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/nmf-test-opencl.dir/build: tests/nmf-test-opencl
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/build

tests/CMakeFiles/nmf-test-opencl.dir/requires: tests/CMakeFiles/nmf-test-opencl.dir/src/nmf.cpp.o.requires
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/requires

tests/CMakeFiles/nmf-test-opencl.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/nmf-test-opencl.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/clean

tests/CMakeFiles/nmf-test-opencl.dir/depend:
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/nmf-test-opencl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/nmf-test-opencl.dir/depend

