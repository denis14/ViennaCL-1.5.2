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
include tests/CMakeFiles/bidiag_test-test-cpu.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/bidiag_test-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/bidiag_test-test-cpu.dir/flags.make

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o: tests/CMakeFiles/bidiag_test-test-cpu.dir/flags.make
tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o: ../tests/src/bidiag_test.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o -c /home/denis/ViennaCL-1.5.2/tests/src/bidiag_test.cpp

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/tests/src/bidiag_test.cpp > CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.i

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/tests/src/bidiag_test.cpp -o CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.s

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.requires:
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.requires

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.provides: tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/bidiag_test-test-cpu.dir/build.make tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.provides.build
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.provides

tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.provides.build: tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o

# Object files for target bidiag_test-test-cpu
bidiag_test__test__cpu_OBJECTS = \
"CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o"

# External object files for target bidiag_test-test-cpu
bidiag_test__test__cpu_EXTERNAL_OBJECTS =

tests/bidiag_test-test-cpu: tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o
tests/bidiag_test-test-cpu: tests/CMakeFiles/bidiag_test-test-cpu.dir/build.make
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_system.so
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tests/bidiag_test-test-cpu: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/bidiag_test-test-cpu: tests/CMakeFiles/bidiag_test-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bidiag_test-test-cpu"
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bidiag_test-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/bidiag_test-test-cpu.dir/build: tests/bidiag_test-test-cpu
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/build

tests/CMakeFiles/bidiag_test-test-cpu.dir/requires: tests/CMakeFiles/bidiag_test-test-cpu.dir/src/bidiag_test.cpp.o.requires
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/requires

tests/CMakeFiles/bidiag_test-test-cpu.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/bidiag_test-test-cpu.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/clean

tests/CMakeFiles/bidiag_test-test-cpu.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/tests /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/tests /home/denis/ViennaCL-1.5.2/build/tests/CMakeFiles/bidiag_test-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/bidiag_test-test-cpu.dir/depend

