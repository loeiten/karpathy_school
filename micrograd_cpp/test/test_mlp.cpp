#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <iostream>    // for operator<<, basic_ostream, endl, cha...
#include <memory>      // for shared_ptr
#include <string>      // for char_traits, string
#include <vector>      // for vector

#include "../include/graph.hpp"  // for Graph
#include "../include/mlp.hpp"    // for Layer
#include "../include/value.hpp"  // for Value

void SimpleMLP() {
  // Create graph where Values reside
  Graph graph{};

  // Create the examples
  std::vector<std::vector<std::shared_ptr<Value>>> x{
      {
          // Example 1
          graph.CreateValue(2.0, "x11").get_shared_ptr(),
          graph.CreateValue(3.0, "x12").get_shared_ptr(),
      },
      {
          // Example 2
          graph.CreateValue(3.0, "x21").get_shared_ptr(),
          graph.CreateValue(-1.0, "x22").get_shared_ptr(),
      },
      {
          // Example 3
          graph.CreateValue(0.5, "x31").get_shared_ptr(),
          graph.CreateValue(1.0, "x32").get_shared_ptr(),
      },
  };

  // Create the ground truth
  std::vector<std::shared_ptr<Value>> ys{
      graph.CreateValue(1.0, "gt1").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt2").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt3").get_shared_ptr(),
  };

  MLP mlp{graph, 2, {2, 2, 1}};

  // Manually change the values to get reproducibility
  auto &parameters = mlp.get_parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    double val = -1.0 + (static_cast<double>(i) / parameters.size()) * 2.0;
    parameters.at(i)->set_data(val);
  }

  auto &loss = mlp.Loss(x, ys);

  std::cout << graph.ReturnDot(loss) << std::endl;
}

void TrainMLP() {
  // Create graph where Values reside
  Graph graph{};

  // Create the examples
  std::vector<std::vector<std::shared_ptr<Value>>> x{
      {
          // Example 1
          graph.CreateValue(2.0, "x11").get_shared_ptr(),
          graph.CreateValue(3.0, "x12").get_shared_ptr(),
      },
      {
          // Example 2
          graph.CreateValue(3.0, "x21").get_shared_ptr(),
          graph.CreateValue(-1.0, "x22").get_shared_ptr(),
      },
      {
          // Example 3
          graph.CreateValue(0.5, "x31").get_shared_ptr(),
          graph.CreateValue(1.0, "x32").get_shared_ptr(),
      },
  };

  // Create the ground truth
  std::vector<std::shared_ptr<Value>> ys{
      graph.CreateValue(1.0, "gt1").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt2").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt3").get_shared_ptr(),
  };

  MLP mlp{graph, 2, {2, 2, 1}};

  // Manually change the values to get reproducibility
  auto &parameters = mlp.get_parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    double val = -1.0 + (static_cast<double>(i) / parameters.size()) * 2.0;
    parameters.at(i)->set_data(val);
  }

  mlp.Train(x, ys, 3);
}

int main(int argc, char **argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("SimpleMLP") == 0) {
    SimpleMLP();
  } else if (std::string(argv[1]).compare("TrainMLP") == 0) {
    TrainMLP();
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
