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
include tests/CMakeFiles/matrix_col_int-test-opencl.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/matrix_col_int-test-opencl.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/matrix_col_int-test-opencl.dir/flags.make

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o: tests/CMakeFiles/matrix_col_int-test-opencl.dir/flags.make
tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o: ../tests/src/matrix_col_int.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o -c /home/denis/ViennaCL-1.5.2/tests/src/matrix_col_int.cpp

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/tests/src/matrix_col_int.cpp > CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.i

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/tests/src/matrix_col_int.cpp -o CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.s

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.requires:
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.requires

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.provides: tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/matrix_col_int-test-opencl.dir/build.make tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.provides.build
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.provides

tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.provides.build: tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o

# Object files for target matrix_col_int-test-opencl
matrix_col_int__test__opencl_OBJECTS = \
"CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o"

# External object files for target matrix_col_int-test-opencl
matrix_col_int__test__opencl_EXTERNAL_OBJECTS =

tests/matrix_col_int-test-opencl: tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o
tests/matrix_col_int-test-opencl: tests/CMakeFiles/matrix_col_int-test-opencl.dir/build.make
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libOpenCL.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_system.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tests/matrix_col_int-test-opencl: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/matrix_col_int-test-opencl: tests/CMakeFiles/matrix_col_int-test-opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable matrix_col_int-test-opencl"
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_col_int-test-opencl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/matrix_col_int-test-opencl.dir/build: tests/matrix_col_int-test-opencl
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/build

tests/CMakeFiles/matrix_col_int-test-opencl.dir/requires: tests/CMakeFiles/matrix_col_int-test-opencl.dir/src/matrix_col_int.cpp.o.requires
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/requires

tests/CMakeFiles/matrix_col_int-test-opencl.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/matrix_col_int-test-opencl.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/clean

tests/CMakeFiles/matrix_col_int-test-opencl.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/tests /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/tests /home/denis/ViennaCL-1.5.2/build/tests/CMakeFiles/matrix_col_int-test-opencl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/matrix_col_int-test-opencl.dir/depend

