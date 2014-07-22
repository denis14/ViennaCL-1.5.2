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
include examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/flags.make

examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/libviennacl-tutorial_generated_libviennacl.cu.o.depend
examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/libviennacl-tutorial_generated_libviennacl.cu.o.cmake
examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o: ../examples/tutorial/libviennacl.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object examples/tutorial/CMakeFiles/libviennacl-tutorial.dir//./libviennacl-tutorial_generated_libviennacl.cu.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir && /usr/bin/cmake -E make_directory /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir//.
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir//./libviennacl-tutorial_generated_libviennacl.cu.o -D generated_cubin_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir//./libviennacl-tutorial_generated_libviennacl.cu.o.cubin.txt -P /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir//libviennacl-tutorial_generated_libviennacl.cu.o.cmake

# Object files for target libviennacl-tutorial
libviennacl__tutorial_OBJECTS =

# External object files for target libviennacl-tutorial
libviennacl__tutorial_EXTERNAL_OBJECTS = \
"/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o"

examples/tutorial/libviennacl-tutorial: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o
examples/tutorial/libviennacl-tutorial: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/build.make
examples/tutorial/libviennacl-tutorial: /usr/lib/x86_64-linux-gnu/libcudart.so
examples/tutorial/libviennacl-tutorial: libviennacl/libviennacl.so
examples/tutorial/libviennacl-tutorial: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/tutorial/libviennacl-tutorial: /usr/lib/x86_64-linux-gnu/libcudart.so
examples/tutorial/libviennacl-tutorial: /usr/lib/x86_64-linux-gnu/libOpenCL.so
examples/tutorial/libviennacl-tutorial: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable libviennacl-tutorial"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libviennacl-tutorial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/build: examples/tutorial/libviennacl-tutorial
.PHONY : examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/build

examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/requires:
.PHONY : examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/requires

examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/libviennacl-tutorial.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/clean

examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/depend: examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/./libviennacl-tutorial_generated_libviennacl.cu.o
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/tutorial /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/tutorial /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/libviennacl-tutorial.dir/depend

