# Use file(GLOB ...) to gather all the op files into the OP_FILES variable
# alternatively one could add the ops manually or make a library out of them
file(GLOB OP_FILES src/ops/*.cpp)

# Set the main classes as a variable
set(MAIN_CLASSES
    src/graph.cpp
    src/value.cpp
    src/module.cpp
    src/neuron.cpp
    src/layer.cpp
    src/mlp.cpp
)

add_executable(main src/main.cpp ${OP_FILES} ${MAIN_CLASSES})

# Test (NOTE: Not all the classes are needed for all the tests)
# Graph
add_executable(test_graph test/test_graph.cpp ${OP_FILES} ${MAIN_CLASSES})
do_test(NAME test_graph NO 1 ARGS "SimpleGraph")
do_test(NAME test_graph NO 2 ARGS "BackProp")
do_test(NAME test_graph NO 3 ARGS "GraphWithTemporaries")
do_test(NAME test_graph NO 4 ARGS "ReuseVariable")
do_test(NAME test_graph NO 5 ARGS "CompoundOps")
do_test(NAME test_graph NO 6 ARGS "TanhSpelledOut")

# Neuron
add_executable(test_neuron test/test_neuron.cpp ${OP_FILES} ${MAIN_CLASSES})
do_test(NAME test_neuron NO 1 ARGS "LinearNeuron")
do_test(NAME test_neuron NO 2 ARGS "NonLinearNeuron")

# Layer
add_executable(test_layer test/test_layer.cpp ${OP_FILES} ${MAIN_CLASSES})
do_test(NAME test_layer NO 1 ARGS "LinearLayer")
do_test(NAME test_layer NO 2 ARGS "NonLinearLayer")

# MLP
add_executable(test_mlp test/test_mlp.cpp ${OP_FILES} ${MAIN_CLASSES})
do_test(NAME test_mlp NO 1 ARGS "SimpleMLP")
do_test(NAME test_mlp NO 2 ARGS "TrainMLP")
