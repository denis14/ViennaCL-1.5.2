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
include tests/CMakeFiles/iterators-test-opencl.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/iterators-test-opencl.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/iterators-test-opencl.dir/flags.make

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o: tests/CMakeFiles/iterators-test-opencl.dir/flags.make
tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o: ../tests/src/iterators.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o -c /home/denis/ViennaCL-1.5.2/tests/src/iterators.cpp

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/tests/src/iterators.cpp > CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.i

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/tests/src/iterators.cpp -o CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.s

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.requires:
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.requires

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.provides: tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/iterators-test-opencl.dir/build.make tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.provides.build
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.provides

tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.provides.build: tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o

# Object files for target iterators-test-opencl
iterators__test__opencl_OBJECTS = \
"CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o"

# External object files for target iterators-test-opencl
iterators__test__opencl_EXTERNAL_OBJECTS =

tests/iterators-test-opencl: tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o
tests/iterators-test-opencl: tests/CMakeFiles/iterators-test-opencl.dir/build.make
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libOpenCL.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_system.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tests/iterators-test-opencl: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/iterators-test-opencl: tests/CMakeFiles/iterators-test-opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable iterators-test-opencl"
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/iterators-test-opencl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/iterators-test-opencl.dir/build: tests/iterators-test-opencl
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/build

tests/CMakeFiles/iterators-test-opencl.dir/requires: tests/CMakeFiles/iterators-test-opencl.dir/src/iterators.cpp.o.requires
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/requires

tests/CMakeFiles/iterators-test-opencl.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/iterators-test-opencl.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/clean

tests/CMakeFiles/iterators-test-opencl.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/tests /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/tests /home/denis/ViennaCL-1.5.2/build/tests/CMakeFiles/iterators-test-opencl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/iterators-test-opencl.dir/depend

