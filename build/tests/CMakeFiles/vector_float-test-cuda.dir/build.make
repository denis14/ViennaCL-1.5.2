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
<<<<<<< HEAD
CMAKE_SOURCE_DIR = /home/denis/Documents/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/denis/Documents/ViennaCL-1.5.2/build
=======
CMAKE_SOURCE_DIR = /home/andi/git/ViennaCL-1.5.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andi/git/ViennaCL-1.5.2/build
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6

# Include any dependencies generated for this target.
include tests/CMakeFiles/vector_float-test-cuda.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/vector_float-test-cuda.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/vector_float-test-cuda.dir/flags.make

tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o: tests/CMakeFiles/vector_float-test-cuda.dir/src/vector_float-test-cuda_generated_vector_float.cu.o.depend
tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o: tests/CMakeFiles/vector_float-test-cuda.dir/src/vector_float-test-cuda_generated_vector_float.cu.o.cmake
tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o: ../tests/src/vector_float.cu
<<<<<<< HEAD
	$(CMAKE_COMMAND) -E cmake_progress_report /home/denis/Documents/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o"
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src && /usr/bin/cmake -E make_directory /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/.
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o -D generated_cubin_file:STRING=/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o.cubin.txt -P /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/vector_float-test-cuda_generated_vector_float.cu.o.cmake
=======
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andi/git/ViennaCL-1.5.2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o"
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src && /usr/bin/cmake -E make_directory /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/.
	cd /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o -D generated_cubin_file:STRING=/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o.cubin.txt -P /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/vector_float-test-cuda_generated_vector_float.cu.o.cmake
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6

# Object files for target vector_float-test-cuda
vector_float__test__cuda_OBJECTS =

# External object files for target vector_float-test-cuda
vector_float__test__cuda_EXTERNAL_OBJECTS = \
<<<<<<< HEAD
"/home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o"
=======
"/home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o"
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6

tests/vector_float-test-cuda: tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o
tests/vector_float-test-cuda: tests/CMakeFiles/vector_float-test-cuda.dir/build.make
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libcudart.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libboost_system.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tests/vector_float-test-cuda: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/vector_float-test-cuda: tests/CMakeFiles/vector_float-test-cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable vector_float-test-cuda"
<<<<<<< HEAD
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_float-test-cuda.dir/link.txt --verbose=$(VERBOSE)
=======
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_float-test-cuda.dir/link.txt --verbose=$(VERBOSE)
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6

# Rule to build all files generated by this target.
tests/CMakeFiles/vector_float-test-cuda.dir/build: tests/vector_float-test-cuda
.PHONY : tests/CMakeFiles/vector_float-test-cuda.dir/build

tests/CMakeFiles/vector_float-test-cuda.dir/requires:
.PHONY : tests/CMakeFiles/vector_float-test-cuda.dir/requires

tests/CMakeFiles/vector_float-test-cuda.dir/clean:
<<<<<<< HEAD
	cd /home/denis/Documents/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/vector_float-test-cuda.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/vector_float-test-cuda.dir/clean

tests/CMakeFiles/vector_float-test-cuda.dir/depend: tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o
	cd /home/denis/Documents/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/denis/Documents/ViennaCL-1.5.2 /home/denis/Documents/ViennaCL-1.5.2/tests /home/denis/Documents/ViennaCL-1.5.2/build /home/denis/Documents/ViennaCL-1.5.2/build/tests /home/denis/Documents/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/DependInfo.cmake --color=$(COLOR)
=======
	cd /home/andi/git/ViennaCL-1.5.2/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/vector_float-test-cuda.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/vector_float-test-cuda.dir/clean

tests/CMakeFiles/vector_float-test-cuda.dir/depend: tests/CMakeFiles/vector_float-test-cuda.dir/src/./vector_float-test-cuda_generated_vector_float.cu.o
	cd /home/andi/git/ViennaCL-1.5.2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andi/git/ViennaCL-1.5.2 /home/andi/git/ViennaCL-1.5.2/tests /home/andi/git/ViennaCL-1.5.2/build /home/andi/git/ViennaCL-1.5.2/build/tests /home/andi/git/ViennaCL-1.5.2/build/tests/CMakeFiles/vector_float-test-cuda.dir/DependInfo.cmake --color=$(COLOR)
>>>>>>> 18ea777fbaf799ee7d33f419621d5d740873a5f6
.PHONY : tests/CMakeFiles/vector_float-test-cuda.dir/depend

