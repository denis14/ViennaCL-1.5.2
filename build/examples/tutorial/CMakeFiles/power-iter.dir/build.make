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
CMAKE_SOURCE_DIR = /home/andi/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include examples/tutorial/CMakeFiles/power-iter.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/power-iter.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/power-iter.dir/flags.make

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o: examples/tutorial/CMakeFiles/power-iter.dir/flags.make
examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o: ../examples/tutorial/power-iter.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o"
	cd /home/andi/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/power-iter.dir/power-iter.cpp.o -c /home/andi/ViennaCL-1.5.2/examples/tutorial/power-iter.cpp

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/power-iter.dir/power-iter.cpp.i"
	cd /home/andi/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/ViennaCL-1.5.2/examples/tutorial/power-iter.cpp > CMakeFiles/power-iter.dir/power-iter.cpp.i

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/power-iter.dir/power-iter.cpp.s"
	cd /home/andi/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/ViennaCL-1.5.2/examples/tutorial/power-iter.cpp -o CMakeFiles/power-iter.dir/power-iter.cpp.s

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.requires:
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.requires

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.provides: examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.requires
	$(MAKE) -f examples/tutorial/CMakeFiles/power-iter.dir/build.make examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.provides.build
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.provides

examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.provides.build: examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o

# Object files for target power-iter
power__iter_OBJECTS = \
"CMakeFiles/power-iter.dir/power-iter.cpp.o"

# External object files for target power-iter
power__iter_EXTERNAL_OBJECTS =

examples/tutorial/power-iter: examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o
examples/tutorial/power-iter: examples/tutorial/CMakeFiles/power-iter.dir/build.make
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_chrono.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_date_time.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_serialization.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_system.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_thread.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libpthread.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libOpenCL.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_chrono.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_date_time.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_serialization.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_system.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libboost_thread.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libpthread.so
examples/tutorial/power-iter: /usr/lib/i386-linux-gnu/libOpenCL.so
examples/tutorial/power-iter: examples/tutorial/CMakeFiles/power-iter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable power-iter"
	cd /home/andi/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/power-iter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/power-iter.dir/build: examples/tutorial/power-iter
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/build

examples/tutorial/CMakeFiles/power-iter.dir/requires: examples/tutorial/CMakeFiles/power-iter.dir/power-iter.cpp.o.requires
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/requires

examples/tutorial/CMakeFiles/power-iter.dir/clean:
	cd /home/andi/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/power-iter.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/clean

examples/tutorial/CMakeFiles/power-iter.dir/depend:
	cd /home/andi/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/ViennaCL-1.5.2 /home/andi/ViennaCL-1.5.2/examples/tutorial /home/andi/ViennaCL-1.5.2/build /home/andi/ViennaCL-1.5.2/build/examples/tutorial /home/andi/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/power-iter.dir/depend

