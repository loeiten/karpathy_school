#include <memory>  // for shared_ptr
#include <vector>  // for vector

#include "../include/graph.hpp"  // for Graph
#include "../include/mlp.hpp"    // for MLP
#include "../include/value.hpp"  // for Value

int main() {
  Graph graph{};

  // Create the examples
  std::vector<std::vector<std::shared_ptr<Value>>> x{
      {// Example 1
       graph.CreateValue(2.0, "x11").get_shared_ptr(),
       graph.CreateValue(3.0, "x12").get_shared_ptr(),
       graph.CreateValue(-1.0, "x13").get_shared_ptr()},
      {// Example 2
       graph.CreateValue(3.0, "x21").get_shared_ptr(),
       graph.CreateValue(-1.0, "x22").get_shared_ptr(),
       graph.CreateValue(0.5, "x23").get_shared_ptr()},
      {// Example 3
       graph.CreateValue(0.5, "x31").get_shared_ptr(),
       graph.CreateValue(1.0, "x32").get_shared_ptr(),
       graph.CreateValue(1.0, "x33").get_shared_ptr()},
      {// Example 4
       graph.CreateValue(1.0, "x41").get_shared_ptr(),
       graph.CreateValue(1.0, "x42").get_shared_ptr(),
       graph.CreateValue(-1.0, "x43").get_shared_ptr()},
  };

  // Create the ground truth
  std::vector<std::shared_ptr<Value>> ys{
      graph.CreateValue(1.0, "gt1").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt2").get_shared_ptr(),
      graph.CreateValue(-1.0, "gt3").get_shared_ptr(),
      graph.CreateValue(1.0, "gt4").get_shared_ptr()};

  MLP mlp{graph, 3, {4, 4, 1}};
  mlp.Train(x, ys, 10);

  return 0;
}
