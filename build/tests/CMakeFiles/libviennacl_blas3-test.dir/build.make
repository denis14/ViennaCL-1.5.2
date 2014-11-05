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
include tests/CMakeFiles/libviennacl_blas3-test.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/libviennacl_blas3-test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/libviennacl_blas3-test.dir/flags.make

tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o: tests/CMakeFiles/libviennacl_blas3-test.dir/src/libviennacl_blas3-test_generated_libviennacl_blas3.cu.o.depend
tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o: tests/CMakeFiles/libviennacl_blas3-test.dir/src/libviennacl_blas3-test_generated_libviennacl_blas3.cu.o.cmake
tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o: ../tests/src/libviennacl_blas3.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src && /usr/bin/cmake -E make_directory /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src/.
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=release -D generated_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o -D generated_cubin_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o.cubin.txt -P /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src/libviennacl_blas3-test_generated_libviennacl_blas3.cu.o.cmake

# Object files for target libviennacl_blas3-test
libviennacl_blas3__test_OBJECTS =

# External object files for target libviennacl_blas3-test
libviennacl_blas3__test_EXTERNAL_OBJECTS = \
"/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o"

tests/libviennacl_blas3-test: tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o
tests/libviennacl_blas3-test: tests/CMakeFiles/libviennacl_blas3-test.dir/build.make
tests/libviennacl_blas3-test: /usr/lib/i386-linux-gnu/libcudart.so
tests/libviennacl_blas3-test: libviennacl/libviennacl.so
tests/libviennacl_blas3-test: /usr/lib/i386-linux-gnu/libOpenCL.so
tests/libviennacl_blas3-test: /usr/lib/i386-linux-gnu/libcudart.so
tests/libviennacl_blas3-test: /usr/lib/i386-linux-gnu/libOpenCL.so
tests/libviennacl_blas3-test: tests/CMakeFiles/libviennacl_blas3-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable libviennacl_blas3-test"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libviennacl_blas3-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/libviennacl_blas3-test.dir/build: tests/libviennacl_blas3-test
.PHONY : tests/CMakeFiles/libviennacl_blas3-test.dir/build

tests/CMakeFiles/libviennacl_blas3-test.dir/requires:
.PHONY : tests/CMakeFiles/libviennacl_blas3-test.dir/requires

tests/CMakeFiles/libviennacl_blas3-test.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/libviennacl_blas3-test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/libviennacl_blas3-test.dir/clean

tests/CMakeFiles/libviennacl_blas3-test.dir/depend: tests/CMakeFiles/libviennacl_blas3-test.dir/src/./libviennacl_blas3-test_generated_libviennacl_blas3.cu.o
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas3-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/libviennacl_blas3-test.dir/depend
