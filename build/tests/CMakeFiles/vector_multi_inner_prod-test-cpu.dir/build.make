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
include tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/flags.make

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/flags.make
tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o: ../tests/src/vector_multi_inner_prod.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/viennacl-dev/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o -c /home/andi/git/viennacl-dev/tests/src/vector_multi_inner_prod.cpp

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.i"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/viennacl-dev/tests/src/vector_multi_inner_prod.cpp > CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.i

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.s"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/viennacl-dev/tests/src/vector_multi_inner_prod.cpp -o CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.s

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.requires:
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.requires

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.provides: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/build.make tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.provides.build
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.provides

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.provides.build: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o

# Object files for target vector_multi_inner_prod-test-cpu
vector_multi_inner_prod__test__cpu_OBJECTS = \
"CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o"

# External object files for target vector_multi_inner_prod-test-cpu
vector_multi_inner_prod__test__cpu_EXTERNAL_OBJECTS =

tests/vector_multi_inner_prod-test-cpu: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o
tests/vector_multi_inner_prod-test-cpu: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/build.make
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libboost_system.so
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/vector_multi_inner_prod-test-cpu: /usr/lib/i386-linux-gnu/libpthread.so
tests/vector_multi_inner_prod-test-cpu: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable vector_multi_inner_prod-test-cpu"
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_multi_inner_prod-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/build: tests/vector_multi_inner_prod-test-cpu
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/build

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/requires: tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/src/vector_multi_inner_prod.cpp.o.requires
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/requires

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/clean:
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/vector_multi_inner_prod-test-cpu.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/clean

tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/depend:
	cd /home/andi/git/viennacl-dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/viennacl-dev /home/andi/git/viennacl-dev/tests /home/andi/git/viennacl-dev/build /home/andi/git/viennacl-dev/build/tests /home/andi/git/viennacl-dev/build/tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/vector_multi_inner_prod-test-cpu.dir/depend

