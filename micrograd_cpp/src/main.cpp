#include <iostream>

#include "../include/print_graph.hpp"
#include "../include/value.hpp"

int main() {
  auto a = Value(2.0);
  auto b = Value(-3.0);
  auto c = Value(10.0);
  auto e = a * b;
  auto d = e + c;

  std::cout << ReturnDot(d) << std::endl;

  return 0;
}
