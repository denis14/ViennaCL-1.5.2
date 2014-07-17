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
include examples/tutorial/CMakeFiles/power-iter-cuda.dir/depend.make

# Include the progress variables for this target.
include examples/tutorial/CMakeFiles/power-iter-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tutorial/CMakeFiles/power-iter-cuda.dir/flags.make

examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o: examples/tutorial/CMakeFiles/power-iter-cuda.dir/power-iter-cuda_generated_power-iter.cu.o.depend
examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o: examples/tutorial/CMakeFiles/power-iter-cuda.dir/power-iter-cuda_generated_power-iter.cu.o.cmake
examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o: ../examples/tutorial/power-iter.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object examples/tutorial/CMakeFiles/power-iter-cuda.dir//./power-iter-cuda_generated_power-iter.cu.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir && /usr/bin/cmake -E make_directory /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir//.
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir//./power-iter-cuda_generated_power-iter.cu.o -D generated_cubin_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir//./power-iter-cuda_generated_power-iter.cu.o.cubin.txt -P /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir//power-iter-cuda_generated_power-iter.cu.o.cmake

# Object files for target power-iter-cuda
power__iter__cuda_OBJECTS =

# External object files for target power-iter-cuda
power__iter__cuda_EXTERNAL_OBJECTS = \
"/home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o"

examples/tutorial/power-iter-cuda: examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o
examples/tutorial/power-iter-cuda: examples/tutorial/CMakeFiles/power-iter-cuda.dir/build.make
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libcudart.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libboost_chrono.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libboost_date_time.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libboost_serialization.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libboost_system.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libboost_thread.so
examples/tutorial/power-iter-cuda: /usr/lib/i386-linux-gnu/libpthread.so
examples/tutorial/power-iter-cuda: examples/tutorial/CMakeFiles/power-iter-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable power-iter-cuda"
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/power-iter-cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tutorial/CMakeFiles/power-iter-cuda.dir/build: examples/tutorial/power-iter-cuda
.PHONY : examples/tutorial/CMakeFiles/power-iter-cuda.dir/build

examples/tutorial/CMakeFiles/power-iter-cuda.dir/requires:
.PHONY : examples/tutorial/CMakeFiles/power-iter-cuda.dir/requires

examples/tutorial/CMakeFiles/power-iter-cuda.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/power-iter-cuda.dir/cmake_clean.cmake
.PHONY : examples/tutorial/CMakeFiles/power-iter-cuda.dir/clean

examples/tutorial/CMakeFiles/power-iter-cuda.dir/depend: examples/tutorial/CMakeFiles/power-iter-cuda.dir/./power-iter-cuda_generated_power-iter.cu.o
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/examples/tutorial /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial /home/andi/git/ViennaCL-1.5.2/build/examples/tutorial/CMakeFiles/power-iter-cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tutorial/CMakeFiles/power-iter-cuda.dir/depend

