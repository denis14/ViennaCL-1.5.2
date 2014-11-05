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
CMAKE_SOURCE_DIR = /home/andi/git/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/scan-test-cpu.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/scan-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/scan-test-cpu.dir/flags.make

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o: tests/CMakeFiles/scan-test-cpu.dir/flags.make
tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o: ../tests/src/scan.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o -c /home/andi/git/ViennaCL-1.5.2/tests/src/scan.cpp

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scan-test-cpu.dir/src/scan.cpp.i"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/ViennaCL-1.5.2/tests/src/scan.cpp > CMakeFiles/scan-test-cpu.dir/src/scan.cpp.i

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scan-test-cpu.dir/src/scan.cpp.s"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/ViennaCL-1.5.2/tests/src/scan.cpp -o CMakeFiles/scan-test-cpu.dir/src/scan.cpp.s

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.requires:
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.requires

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.provides: tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/scan-test-cpu.dir/build.make tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.provides.build
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.provides

tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.provides.build: tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o

# Object files for target scan-test-cpu
scan__test__cpu_OBJECTS = \
"CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o"

# External object files for target scan-test-cpu
scan__test__cpu_EXTERNAL_OBJECTS =

tests/scan-test-cpu: tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o
tests/scan-test-cpu: tests/CMakeFiles/scan-test-cpu.dir/build.make
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libboost_system.so
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/scan-test-cpu: /usr/lib/i386-linux-gnu/libpthread.so
tests/scan-test-cpu: tests/CMakeFiles/scan-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable scan-test-cpu"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/scan-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/scan-test-cpu.dir/build: tests/scan-test-cpu
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/build

tests/CMakeFiles/scan-test-cpu.dir/requires: tests/CMakeFiles/scan-test-cpu.dir/src/scan.cpp.o.requires
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/requires

tests/CMakeFiles/scan-test-cpu.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/scan-test-cpu.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/clean

tests/CMakeFiles/scan-test-cpu.dir/depend:
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/scan-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/scan-test-cpu.dir/depend
