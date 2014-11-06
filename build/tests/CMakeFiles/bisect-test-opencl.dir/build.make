# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_SOURCE_DIR = /home/andi/git/viennacl-dev

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/viennacl-dev/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/bisect-test-opencl.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/bisect-test-opencl.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/bisect-test-opencl.dir/flags.make

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o: tests/CMakeFiles/bisect-test-opencl.dir/flags.make
tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o: ../tests/src/bisect.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/viennacl-dev/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o -c /home/andi/git/viennacl-dev/tests/src/bisect.cpp

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.i"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/viennacl-dev/tests/src/bisect.cpp > CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.i

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.s"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/viennacl-dev/tests/src/bisect.cpp -o CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.s

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.requires:
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.requires

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.provides: tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/bisect-test-opencl.dir/build.make tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.provides.build
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.provides

tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.provides.build: tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o

# Object files for target bisect-test-opencl
bisect__test__opencl_OBJECTS = \
"CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o"

# External object files for target bisect-test-opencl
bisect__test__opencl_EXTERNAL_OBJECTS =

tests/bisect-test-opencl: tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o
tests/bisect-test-opencl: tests/CMakeFiles/bisect-test-opencl.dir/build.make
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libOpenCL.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libboost_system.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/bisect-test-opencl: /usr/lib/i386-linux-gnu/libpthread.so
tests/bisect-test-opencl: tests/CMakeFiles/bisect-test-opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bisect-test-opencl"
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bisect-test-opencl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/bisect-test-opencl.dir/build: tests/bisect-test-opencl
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/build

tests/CMakeFiles/bisect-test-opencl.dir/requires: tests/CMakeFiles/bisect-test-opencl.dir/src/bisect.cpp.o.requires
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/requires

tests/CMakeFiles/bisect-test-opencl.dir/clean:
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/bisect-test-opencl.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/clean

tests/CMakeFiles/bisect-test-opencl.dir/depend:
	cd /home/andi/git/viennacl-dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/viennacl-dev /home/andi/git/viennacl-dev/tests /home/andi/git/viennacl-dev/build /home/andi/git/viennacl-dev/build/tests /home/andi/git/viennacl-dev/build/tests/CMakeFiles/bisect-test-opencl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/bisect-test-opencl.dir/depend

