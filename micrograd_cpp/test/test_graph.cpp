#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <iostream>    // for operator<<, basic_ostream, endl
#include <string>      // for char_traits, string

#include "../include/graph.hpp"  // for ReturnDot
#include "../include/value.hpp"  // for Value

/*
FIXME:
I have a graph like so

t1        t4
  \        \
    + - t3 * - t5 - cos - t6
  /
t2

I want the user to be able to write something like this
((t1 + t2) * t4).cos()

hence
t3 = t1 + t2
t5 = t3 * t4

If we ask how much t1 contribute to t6, we get

d/dt1 t6 = d/dt1 t6( t5(t4, t3(t2, t1)) )
         = dt6/dt6 * dt6/dt5 * dt5/dt3 * dt3/dt1

This means that the gradient of t1 will be dependent on t3, said
differently t3 will modify t1

Questions:
- Where should the ownership be of the t nodes
  - Graph should own it, but how will I be able to do chaining then?
    Comment: One could use dereference and change to
    (mul(add(*t1, *t2), *t4))->cos()
    (*(*t1 + *t2) * *t4)->cos()

Pattern called: PIMPL (pointer to implementation)
ValueImpl  - class having all the actual data
Value - shell containing overloaded ops, and a weak_ptr to ValueImpl
Graph - owns the ValueImpl and stores it as a shared_ptr in some container
End user should not have access to ValueImpl
Op - class which has fwd and bwd and all inputs and outputs
*/

void SimpleGraph() {
  // Create graph where Values reside
  auto graph = Graph();
  auto a = graph.CreateValue(2.0, "a");
  auto b = graph.CreateValue(-3.0, "b");
  auto c = graph.CreateValue(10.0, "c");
  auto e = a * b;
  e.set_label("e");
  auto d = e + c;
  d.set_label("d");

  std::cout << graph.ReturnDot(d) << std::endl;
}

void BackProp() {
  // Create graph where Values reside
  auto graph = Graph();
  // Inputs x1, x2
  auto x1 = graph.CreateValue(2.0, "x1");
  auto x2 = graph.CreateValue(0.0, "x2");
  // Weights w1, w2
  auto w1 = graph.CreateValue(-3.0, "w1");
  auto w2 = graph.CreateValue(1.0, "w2");
  // Bias of the neuron
  auto b = graph.CreateValue(6.8813735870195432, "b");
  // x1w1 + x2w2 + b
  auto x1w1 = x1 * w1;
  x1w1.set_label("x1w1");
  auto x2w2 = x2 * w2;
  x2w2.set_label("x2w2");
  auto x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2.set_label("x1w1 + x2w2");
  auto n = x1w1x2w2 + b;
  n.set_label("n");
  // Output
  auto o = tanh(n);
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

void ReuseVariable() {
  // Create graph where Values reside
  auto graph = Graph();
  auto a = graph.CreateValue(-2.0, "a");
  auto b = graph.CreateValue(3.0, "b");
  auto d = a * b;
  d.set_label("d");
  auto e = a + b;
  e.set_label("e");
  auto f = d * e;
  f.set_label("f");

  f.Backward();

  std::cout << graph.ReturnDot(f) << std::endl;
}

void TanhSpelledOut() {
  // Create graph where Values reside
  auto graph = Graph();
  // Inputs x1, x2
  auto x1 = graph.CreateValue(2.0, "x1");
  auto x2 = graph.CreateValue(0.0, "x2");
  // Weights w1, w2
  auto w1 = graph.CreateValue(-3.0, "w1");
  auto w2 = graph.CreateValue(1.0, "w2");
  // Bias of the neuron
  auto b = graph.CreateValue(6.8813735870195432, "b");
  // x1w1 + x2w2 + b
  auto x1w1 = x1 * w1;
  x1w1.set_label("x1w1");
  auto x2w2 = x2 * w2;
  x2w2.set_label("x2w2");
  auto x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2.set_label("x1w1 + x2w2");
  auto n = x1w1x2w2 + b;
  n.set_label("n");
  // Output
  auto e = exp(2.0f*n);
  auto o = (e-1.0f)/(e+1.0f);
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

int main(int argc, char** argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("SimpleGraph") == 0) {
    SimpleGraph();
  } else if (std::string(argv[1]).compare("BackProp") == 0) {
    BackProp();
  } else if (std::string(argv[1]).compare("ReuseVariable") == 0) {
    ReuseVariable();
  } else if (std::string(argv[1]).compare("TanhSpelledOut") == 0) {
    TanhSpelledOut();
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
