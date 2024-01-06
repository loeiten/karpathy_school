#include <iostream>  // for char_traits, operator<<, basic_ostream

#include "../include/graph.hpp"  // for Graph
#include "../include/value.hpp"  // for Value, exp, operator*

int main() {
  auto graph = Graph();
  auto n = graph.CreateValue(1.0f, "n");
  auto tmp = (2.0f * n);
  auto e = exp(tmp);
  e.set_label("e");

  e.Backward();

  std::cout << graph.ReturnDot(e) << std::endl;

  return 0;
}
