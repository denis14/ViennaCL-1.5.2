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
include tests/CMakeFiles/scheduler_vector-test-cpu.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/scheduler_vector-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/scheduler_vector-test-cpu.dir/flags.make

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o: tests/CMakeFiles/scheduler_vector-test-cpu.dir/flags.make
tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o: ../tests/src/scheduler_vector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/viennacl-dev/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o -c /home/andi/git/viennacl-dev/tests/src/scheduler_vector.cpp

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.i"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/viennacl-dev/tests/src/scheduler_vector.cpp > CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.i

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.s"
	cd /home/andi/git/viennacl-dev/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/viennacl-dev/tests/src/scheduler_vector.cpp -o CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.s

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.requires:
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.requires

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.provides: tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/scheduler_vector-test-cpu.dir/build.make tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.provides.build
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.provides

tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.provides.build: tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o

# Object files for target scheduler_vector-test-cpu
scheduler_vector__test__cpu_OBJECTS = \
"CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o"

# External object files for target scheduler_vector-test-cpu
scheduler_vector__test__cpu_EXTERNAL_OBJECTS =

tests/scheduler_vector-test-cpu: tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o
tests/scheduler_vector-test-cpu: tests/CMakeFiles/scheduler_vector-test-cpu.dir/build.make
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libboost_system.so
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/scheduler_vector-test-cpu: /usr/lib/i386-linux-gnu/libpthread.so
tests/scheduler_vector-test-cpu: tests/CMakeFiles/scheduler_vector-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable scheduler_vector-test-cpu"
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/scheduler_vector-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/scheduler_vector-test-cpu.dir/build: tests/scheduler_vector-test-cpu
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/build

tests/CMakeFiles/scheduler_vector-test-cpu.dir/requires: tests/CMakeFiles/scheduler_vector-test-cpu.dir/src/scheduler_vector.cpp.o.requires
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/requires

tests/CMakeFiles/scheduler_vector-test-cpu.dir/clean:
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/scheduler_vector-test-cpu.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/clean

tests/CMakeFiles/scheduler_vector-test-cpu.dir/depend:
	cd /home/andi/git/viennacl-dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/viennacl-dev /home/andi/git/viennacl-dev/tests /home/andi/git/viennacl-dev/build /home/andi/git/viennacl-dev/build/tests /home/andi/git/viennacl-dev/build/tests/CMakeFiles/scheduler_vector-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/scheduler_vector-test-cpu.dir/depend

