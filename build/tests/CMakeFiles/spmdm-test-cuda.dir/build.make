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
include tests/CMakeFiles/spmdm-test-cuda.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/spmdm-test-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/spmdm-test-cuda.dir/flags.make

tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o: tests/CMakeFiles/spmdm-test-cuda.dir/src/spmdm-test-cuda_generated_spmdm.cu.o.depend
tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o: tests/CMakeFiles/spmdm-test-cuda.dir/src/spmdm-test-cuda_generated_spmdm.cu.o.cmake
tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o: ../tests/src/spmdm.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src && /usr/bin/cmake -E make_directory /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src/.
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=release -D generated_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o -D generated_cubin_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o.cubin.txt -P /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src/spmdm-test-cuda_generated_spmdm.cu.o.cmake

# Object files for target spmdm-test-cuda
spmdm__test__cuda_OBJECTS =

# External object files for target spmdm-test-cuda
spmdm__test__cuda_EXTERNAL_OBJECTS = \
"/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o"

tests/spmdm-test-cuda: tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o
tests/spmdm-test-cuda: tests/CMakeFiles/spmdm-test-cuda.dir/build.make
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libcudart.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libboost_chrono.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libboost_date_time.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libboost_serialization.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libboost_system.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libboost_thread.so
tests/spmdm-test-cuda: /usr/lib/i386-linux-gnu/libpthread.so
tests/spmdm-test-cuda: tests/CMakeFiles/spmdm-test-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable spmdm-test-cuda"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spmdm-test-cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/spmdm-test-cuda.dir/build: tests/spmdm-test-cuda
.PHONY : tests/CMakeFiles/spmdm-test-cuda.dir/build

tests/CMakeFiles/spmdm-test-cuda.dir/requires:
.PHONY : tests/CMakeFiles/spmdm-test-cuda.dir/requires

tests/CMakeFiles/spmdm-test-cuda.dir/clean:
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/spmdm-test-cuda.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/spmdm-test-cuda.dir/clean

tests/CMakeFiles/spmdm-test-cuda.dir/depend: tests/CMakeFiles/spmdm-test-cuda.dir/src/./spmdm-test-cuda_generated_spmdm.cu.o
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/spmdm-test-cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/spmdm-test-cuda.dir/depend

