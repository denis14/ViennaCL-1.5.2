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
include examples/tutorial/CMakeFiles/structured-matrices.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/structured-matrices.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/structured-matrices.dir/flags.make

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o: examples/tutorial/CMakeFiles/structured-matrices.dir/flags.make
examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o: ../examples/tutorial/structured-matrices.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o -c /home/denis/ViennaCL-1.5.2/examples/tutorial/structured-matrices.cpp

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/structured-matrices.dir/structured-matrices.cpp.i"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/denis/ViennaCL-1.5.2/examples/tutorial/structured-matrices.cpp > CMakeFiles/structured-matrices.dir/structured-matrices.cpp.i

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/structured-matrices.dir/structured-matrices.cpp.s"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/denis/ViennaCL-1.5.2/examples/tutorial/structured-matrices.cpp -o CMakeFiles/structured-matrices.dir/structured-matrices.cpp.s

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.requires:
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.requires

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.provides: examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.requires
	$(MAKE) -f examples/tutorial/CMakeFiles/structured-matrices.dir/build.make examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.provides.build
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.provides

examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.provides.build: examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o

# Object files for target structured-matrices
structured__matrices_OBJECTS = \
"CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o"

# External object files for target structured-matrices
structured__matrices_EXTERNAL_OBJECTS =

examples/tutorial/structured-matrices: examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o
examples/tutorial/structured-matrices: examples/tutorial/CMakeFiles/structured-matrices.dir/build.make
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/tutorial/structured-matrices: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/tutorial/structured-matrices: examples/tutorial/CMakeFiles/structured-matrices.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable structured-matrices"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/structured-matrices.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/structured-matrices.dir/build: examples/tutorial/structured-matrices
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/build

examples/tutorial/CMakeFiles/structured-matrices.dir/requires: examples/tutorial/CMakeFiles/structured-matrices.dir/structured-matrices.cpp.o.requires
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/requires

examples/tutorial/CMakeFiles/structured-matrices.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/structured-matrices.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/clean

examples/tutorial/CMakeFiles/structured-matrices.dir/depend:
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/tutorial /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/tutorial /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/structured-matrices.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/structured-matrices.dir/depend

