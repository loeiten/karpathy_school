add_executable(main src/main.cpp src/value.cpp src/print_graph.cpp)

# Test
add_executable(test_graph test/test_graph.cpp src/value.cpp src/print_graph.cpp)
do_test(NAME test_graph NO 1 ARGS "SimpleGraph")
do_test(NAME test_graph NO 2 ARGS "BackProp")
