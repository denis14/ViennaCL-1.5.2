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
CMAKE_BINARY_DIR = /home/denis/Documents/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/matrix_vector-test-cpu.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/matrix_vector-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/matrix_vector-test-cpu.dir/flags.make

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o: tests/CMakeFiles/matrix_vector-test-cpu.dir/flags.make
tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o: ../tests/src/matrix_vector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/Documents/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o -c /home/denis/Documents/ViennaCL-1.5.2/tests/src/matrix_vector.cpp

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.i"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/Documents/ViennaCL-1.5.2/tests/src/matrix_vector.cpp > CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.i

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.s"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/Documents/ViennaCL-1.5.2/tests/src/matrix_vector.cpp -o CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.s

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.requires:
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.requires

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.provides: tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/matrix_vector-test-cpu.dir/build.make tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.provides.build
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.provides

tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.provides.build: tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o

# Object files for target matrix_vector-test-cpu
matrix_vector__test__cpu_OBJECTS = \
"CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o"

# External object files for target matrix_vector-test-cpu
matrix_vector__test__cpu_EXTERNAL_OBJECTS =

tests/matrix_vector-test-cpu: tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o
tests/matrix_vector-test-cpu: tests/CMakeFiles/matrix_vector-test-cpu.dir/build.make
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_system.so
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tests/matrix_vector-test-cpu: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/matrix_vector-test-cpu: tests/CMakeFiles/matrix_vector-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable matrix_vector-test-cpu"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_vector-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/matrix_vector-test-cpu.dir/build: tests/matrix_vector-test-cpu
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/build

tests/CMakeFiles/matrix_vector-test-cpu.dir/requires: tests/CMakeFiles/matrix_vector-test-cpu.dir/src/matrix_vector.cpp.o.requires
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/requires

tests/CMakeFiles/matrix_vector-test-cpu.dir/clean:
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/matrix_vector-test-cpu.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/clean

tests/CMakeFiles/matrix_vector-test-cpu.dir/depend:
	cd /home/denis/Documents/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/Documents/ViennaCL-1.5.2 /home/denis/Documents/ViennaCL-1.5.2/tests /home/denis/Documents/ViennaCL-1.5.2/build /home/denis/Documents/ViennaCL-1.5.2/build/tests /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/matrix_vector-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/matrix_vector-test-cpu.dir/depend

