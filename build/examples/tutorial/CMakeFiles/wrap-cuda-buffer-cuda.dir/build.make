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
include examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/flags.make

examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o.depend
examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o.cmake
examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o: ../examples/tutorial/wrap-cuda-buffer.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir//./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir && /usr/bin/cmake -E make_directory /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir//.
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir//./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o -D generated_cubin_file:STRING=/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir//./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o.cubin.txt -P /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir//wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o.cmake

# Object files for target wrap-cuda-buffer-cuda
wrap__cuda__buffer__cuda_OBJECTS =

# External object files for target wrap-cuda-buffer-cuda
wrap__cuda__buffer__cuda_EXTERNAL_OBJECTS = \
"/home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o"

examples/tutorial/wrap-cuda-buffer-cuda: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o
examples/tutorial/wrap-cuda-buffer-cuda: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/build.make
examples/tutorial/wrap-cuda-buffer-cuda: /usr/lib/x86_64-linux-gnu/libcudart.so
examples/tutorial/wrap-cuda-buffer-cuda: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable wrap-cuda-buffer-cuda"
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wrap-cuda-buffer-cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/build: examples/tutorial/wrap-cuda-buffer-cuda
.PHONY : examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/build

examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/requires:
.PHONY : examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/requires

examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/clean:
	cd /home/denis/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/wrap-cuda-buffer-cuda.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/clean

examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/depend: examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/./wrap-cuda-buffer-cuda_generated_wrap-cuda-buffer.cu.o
	cd /home/denis/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/ViennaCL-1.5.2 /home/denis/ViennaCL-1.5.2/examples/tutorial /home/denis/ViennaCL-1.5.2/build /home/denis/ViennaCL-1.5.2/build/examples/tutorial /home/denis/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/wrap-cuda-buffer-cuda.dir/depend

