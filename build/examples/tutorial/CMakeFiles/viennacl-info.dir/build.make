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
CMAKE_SOURCE_DIR = /home/andi/git/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/ViennaCL-1.5.2/build

# Include any dependencies generated for this target.
include examples/tutorial/CMakeFiles/viennacl-info.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/viennacl-info.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/viennacl-info.dir/flags.make

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o: examples/tutorial/CMakeFiles/viennacl-info.dir/flags.make
examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o: ../examples/tutorial/viennacl-info.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o -c /home/andi/git/ViennaCL-1.5.2/examples/tutorial/viennacl-info.cpp

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/viennacl-info.dir/viennacl-info.cpp.i"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andi/git/ViennaCL-1.5.2/examples/tutorial/viennacl-info.cpp > CMakeFiles/viennacl-info.dir/viennacl-info.cpp.i

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/viennacl-info.dir/viennacl-info.cpp.s"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andi/git/ViennaCL-1.5.2/examples/tutorial/viennacl-info.cpp -o CMakeFiles/viennacl-info.dir/viennacl-info.cpp.s

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.requires:
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.requires

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.provides: examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.requires
	$(MAKE) -f examples/tutorial/CMakeFiles/viennacl-info.dir/build.make examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.provides.build
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.provides

examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.provides.build: examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o

# Object files for target viennacl-info
viennacl__info_OBJECTS = \
"CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o"

# External object files for target viennacl-info
viennacl__info_EXTERNAL_OBJECTS =

examples/tutorial/viennacl-info: examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o
examples/tutorial/viennacl-info: examples/tutorial/CMakeFiles/viennacl-info.dir/build.make
examples/tutorial/viennacl-info: /usr/lib/i386-linux-gnu/libOpenCL.so
examples/tutorial/viennacl-info: examples/tutorial/CMakeFiles/viennacl-info.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable viennacl-info"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/viennacl-info.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/viennacl-info.dir/build: examples/tutorial/viennacl-info
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/build

examples/tutorial/CMakeFiles/viennacl-info.dir/requires: examples/tutorial/CMakeFiles/viennacl-info.dir/viennacl-info.cpp.o.requires
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/requires

examples/tutorial/CMakeFiles/viennacl-info.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/viennacl-info.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/clean

examples/tutorial/CMakeFiles/viennacl-info.dir/depend:
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/examples/tutorial /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/viennacl-info.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/viennacl-info.dir/depend

