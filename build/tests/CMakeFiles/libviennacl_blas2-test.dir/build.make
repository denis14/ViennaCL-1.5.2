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
include tests/CMakeFiles/libviennacl_blas2-test.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/libviennacl_blas2-test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/libviennacl_blas2-test.dir/flags.make

tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o: tests/CMakeFiles/libviennacl_blas2-test.dir/src/libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.depend
tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o: tests/CMakeFiles/libviennacl_blas2-test.dir/src/libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.cmake
tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o: ../tests/src/libviennacl_blas2.cu
<<<<<<< HEAD
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/Documents/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src && /usr/bin/cmake -E make_directory /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/.
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o -D generated_cubin_file:STRING=/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.cubin.txt -P /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.cmake
=======
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src && /usr/bin/cmake -E make_directory /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/.
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o -D generated_cubin_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.cubin.txt -P /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/libviennacl_blas2-test_generated_libviennacl_blas2.cu.o.cmake
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6

# Object files for target libviennacl_blas2-test
libviennacl_blas2__test_OBJECTS =

# External object files for target libviennacl_blas2-test
libviennacl_blas2__test_EXTERNAL_OBJECTS = \
<<<<<<< HEAD
"/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o"

tests/libviennacl_blas2-test: tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o
tests/libviennacl_blas2-test: tests/CMakeFiles/libviennacl_blas2-test.dir/build.make
tests/libviennacl_blas2-test: /usr/lib/x86_64-linux-gnu/libcudart.so
tests/libviennacl_blas2-test: libviennacl/libviennacl.so
tests/libviennacl_blas2-test: /usr/lib/x86_64-linux-gnu/libOpenCL.so
tests/libviennacl_blas2-test: /usr/lib/x86_64-linux-gnu/libcudart.so
tests/libviennacl_blas2-test: /usr/lib/x86_64-linux-gnu/libOpenCL.so
=======
"/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o"

tests/libviennacl_blas2-test: tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o
tests/libviennacl_blas2-test: tests/CMakeFiles/libviennacl_blas2-test.dir/build.make
tests/libviennacl_blas2-test: /usr/lib/i386-linux-gnu/libcudart.so
tests/libviennacl_blas2-test: libviennacl/libviennacl.so
tests/libviennacl_blas2-test: /usr/lib/i386-linux-gnu/libOpenCL.so
tests/libviennacl_blas2-test: /usr/lib/i386-linux-gnu/libcudart.so
tests/libviennacl_blas2-test: /usr/lib/i386-linux-gnu/libOpenCL.so
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6
tests/libviennacl_blas2-test: tests/CMakeFiles/libviennacl_blas2-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable libviennacl_blas2-test"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libviennacl_blas2-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/libviennacl_blas2-test.dir/build: tests/libviennacl_blas2-test
.PHONY : tests/CMakeFiles/libviennacl_blas2-test.dir/build

tests/CMakeFiles/libviennacl_blas2-test.dir/requires:
.PHONY : tests/CMakeFiles/libviennacl_blas2-test.dir/requires

tests/CMakeFiles/libviennacl_blas2-test.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/libviennacl_blas2-test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/libviennacl_blas2-test.dir/clean

tests/CMakeFiles/libviennacl_blas2-test.dir/depend: tests/CMakeFiles/libviennacl_blas2-test.dir/src/./libviennacl_blas2-test_generated_libviennacl_blas2.cu.o
<<<<<<< HEAD
	cd /home/denis/Documents/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/Documents/ViennaCL-1.5.2 /home/denis/Documents/ViennaCL-1.5.2/tests /home/denis/Documents/ViennaCL-1.5.2/build /home/denis/Documents/ViennaCL-1.5.2/build/tests /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/DependInfo.cmake --color=$(COLOR)
=======
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/libviennacl_blas2-test.dir/DependInfo.cmake --color=$(COLOR)
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6
.PHONY : tests/CMakeFiles/libviennacl_blas2-test.dir/depend

