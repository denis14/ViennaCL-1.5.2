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
CMAKE_SOURCE_DIR = /home/andi/git/viennacl-dev

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/viennacl-dev/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/matrix_col_double-test-cuda.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/matrix_col_double-test-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/matrix_col_double-test-cuda.dir/flags.make

tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o: tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/matrix_col_double-test-cuda_generated_matrix_col_double.cu.o.depend
tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o: tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/matrix_col_double-test-cuda_generated_matrix_col_double.cu.o.cmake
tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o: ../tests/src/matrix_col_double.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/viennacl-dev/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o"
	cd /home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src && /usr/bin/cmake -E make_directory /home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/.
	cd /home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o -D generated_cubin_file:STRING=/home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o.cubin.txt -P /home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/matrix_col_double-test-cuda_generated_matrix_col_double.cu.o.cmake

# Object files for target matrix_col_double-test-cuda
matrix_col_double__test__cuda_OBJECTS =

# External object files for target matrix_col_double-test-cuda
matrix_col_double__test__cuda_EXTERNAL_OBJECTS = \
"/home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o"

tests/matrix_col_double-test-cuda: tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o
tests/matrix_col_double-test-cuda: tests/CMakeFiles/matrix_col_double-test-cuda.dir/build.make
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libcudart.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libboost_system.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/matrix_col_double-test-cuda: /usr/lib/i386-linux-gnu/libpthread.so
tests/matrix_col_double-test-cuda: tests/CMakeFiles/matrix_col_double-test-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable matrix_col_double-test-cuda"
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_col_double-test-cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/matrix_col_double-test-cuda.dir/build: tests/matrix_col_double-test-cuda
.PHONY : tests/CMakeFiles/matrix_col_double-test-cuda.dir/build

tests/CMakeFiles/matrix_col_double-test-cuda.dir/requires:
.PHONY : tests/CMakeFiles/matrix_col_double-test-cuda.dir/requires

tests/CMakeFiles/matrix_col_double-test-cuda.dir/clean:
	cd /home/andi/git/viennacl-dev/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/matrix_col_double-test-cuda.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/matrix_col_double-test-cuda.dir/clean

tests/CMakeFiles/matrix_col_double-test-cuda.dir/depend: tests/CMakeFiles/matrix_col_double-test-cuda.dir/src/./matrix_col_double-test-cuda_generated_matrix_col_double.cu.o
	cd /home/andi/git/viennacl-dev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/viennacl-dev /home/andi/git/viennacl-dev/tests /home/andi/git/viennacl-dev/build /home/andi/git/viennacl-dev/build/tests /home/andi/git/viennacl-dev/build/tests/CMakeFiles/matrix_col_double-test-cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/matrix_col_double-test-cuda.dir/depend

