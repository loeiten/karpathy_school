# Build with
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja ..  # Creates make files
# cmake --build . -j 4 # Calls make using 4 threads
# ctest -V  # Calls the tests

# See also
# https://cliutils.gitlab.io/modern-cmake/
# https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html
# http://www.aosabook.org/en/cmake.html

cmake_minimum_required(VERSION 3.15)

# Set the project name
# Set the version of the project
project(KARPATHY_SCHOOL VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Turn on compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")
# NOTE: How to set the C_FLAGS are opinoinated, see for example
# https://www.reddit.com/r/cpp_questions/comments/taxyut/what_is_the_best_way_to_set_compiler_flags_in_a/
# List obtained from
# https://stackoverflow.com/questions/3375697/what-are-the-useful-gcc-flags-for-c
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Werror")

# Specify verbose output
# You can also get the commands called by
# cmake --build . -v
# set(CMAKE_VERBOSE_MAKEFILE on)

# Set options
option(IWYU "Build with Include What You Use" OFF)  # Prefer pre-commit
# Sanitizers, see https://github.com/google/sanitizers for details
option(ASAN "Use address sanitizer" OFF)
option(UBSAN "Use undefined behavior sanitizer" OFF)
option(MSAN "Use memory sanitizer" OFF)
option(TSAN "Use thread sanitizer" OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ASAN ON)
    set(UBSAN ON)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

if(IWYU)
   set(CMAKE_C_INCLUDE_WHAT_YOU_USE "include-what-you-use;-Xiwyu;--error;-Xiwyu;--mapping_file=${CMAKE_CURRENT_LIST_DIR}/.linters/iwyu.imp")
endif(IWYU)

# include(CTest) automatically calls enable_testing()
# enable_testing()
include(CTest)


# Define a function to check the output of a test
function(do_test)
   # For details of function parameters, see
   # https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
   cmake_parse_arguments(
       TEST_ARGS            # Prefix of output variables
       ""                   # List of names of the boolean arguments (only defined ones will be true)
       "NAME;NO"            # List of names of mono-valued arguments
       "ARGS"               # List of names of multi-valued arguments (output variables are lists)
       ${ARGN}              # Arguments of the function to parse, here we take the all original ones
   )
   # NOTE: If unparsed arguments remains here, they can be found in the
   #       variable TEST_ARGS_UNPARSED_ARGUMENTS

   # Error handling
   if(NOT TEST_ARGS_NAME)
       message(FATAL_ERROR "You must provide a name to the executable")
   endif(NOT TEST_ARGS_NAME)
   if(NOT TEST_ARGS_NO)
       message(FATAL_ERROR "You must provide a test number")
   endif(NOT TEST_ARGS_NO)

   # Find path relative to this
   file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR})
   string(REPLACE "/" "_" rel_path ${rel_path})
   set(full_name ${rel_path}_${TEST_ARGS_NAME}_${TEST_ARGS_NO})
   # Run the test and compare it with a file
   add_test(NAME ${full_name} COMMAND bash -c "diff <(${CMAKE_CURRENT_BINARY_DIR}/${TEST_ARGS_NAME} ${TEST_ARGS_ARGS}) ${CMAKE_CURRENT_LIST_DIR}/test/data/${TEST_ARGS_ARGS}.txt")
endfunction(do_test)


# Enable sanitizers
if(ASAN)
   add_compile_options(-fsanitize=address)
   add_link_options(-fsanitize=address)
endif(ASAN)
if(UBSAN)
   add_compile_options(-fsanitize=undefined)
   add_link_options(-fsanitize=undefined)
endif(UBSAN)
if(MSAN)
   add_compile_options(-fsanitize=memory)
   add_link_options(-fsanitize=memory)
endif(MSAN)
if(TSAN)
   add_compile_options(-fsanitize=thread)
   add_link_options(-fsanitize=thread)
endif(TSAN)

add_subdirectory(micrograd_cpp)
