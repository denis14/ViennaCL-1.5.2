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
include examples/tutorial/CMakeFiles/blas1-cuda.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/blas1-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/blas1-cuda.dir/flags.make

examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o: examples/tutorial/CMakeFiles/blas1-cuda.dir/blas1-cuda_generated_blas1.cu.o.depend
examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o: examples/tutorial/CMakeFiles/blas1-cuda.dir/blas1-cuda_generated_blas1.cu.o.cmake
examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o: ../examples/tutorial/blas1.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object examples/tutorial/CMakeFiles/blas1-cuda.dir//./blas1-cuda_generated_blas1.cu.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir && /usr/bin/cmake -E make_directory /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir//.
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir//./blas1-cuda_generated_blas1.cu.o -D generated_cubin_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir//./blas1-cuda_generated_blas1.cu.o.cubin.txt -P /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir//blas1-cuda_generated_blas1.cu.o.cmake

# Object files for target blas1-cuda
blas1__cuda_OBJECTS =

# External object files for target blas1-cuda
blas1__cuda_EXTERNAL_OBJECTS = \
"/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o"

examples/tutorial/blas1-cuda: examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o
examples/tutorial/blas1-cuda: examples/tutorial/CMakeFiles/blas1-cuda.dir/build.make
examples/tutorial/blas1-cuda: /usr/lib/x86_64-linux-gnu/libcudart.so
examples/tutorial/blas1-cuda: examples/tutorial/CMakeFiles/blas1-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable blas1-cuda"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blas1-cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/blas1-cuda.dir/build: examples/tutorial/blas1-cuda
.PHONY : examples/tutorial/CMakeFiles/blas1-cuda.dir/build

examples/tutorial/CMakeFiles/blas1-cuda.dir/requires:
.PHONY : examples/tutorial/CMakeFiles/blas1-cuda.dir/requires

examples/tutorial/CMakeFiles/blas1-cuda.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/blas1-cuda.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/blas1-cuda.dir/clean

examples/tutorial/CMakeFiles/blas1-cuda.dir/depend: examples/tutorial/CMakeFiles/blas1-cuda.dir/./blas1-cuda_generated_blas1.cu.o
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/tutorial /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/tutorial /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/blas1-cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/blas1-cuda.dir/depend

