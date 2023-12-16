#ifndef MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_GRAPH_HPP_

#include <memory>
#include <set>            // for set
#include <string>         // for string
#include <unordered_set>  // for set
#include <utility>        // for pair

class Value;

class Graph {
 public:
  // NOTE: We should return a reference to a value so that we can do
  // t1 = graph.CreateValue(2.0);
  // t2 = graph.CreateValue(3.0);
  // t3 = t1 + t2;
  Value& CreateValue(const double &value);
  Value& CreateValue(const double &value, const std::string &label);
  // FIXME: Fix these stray dogs
  // NOTE: We use set here as we want the printed result to be reproducible
  void Trace(const Value &value, std::set<const Value *> *nodes,
             std::set<std::pair<const Value *, const Value *>> *edges);

  std::string ReturnDot(const Value &root);

 private:
  std::unordered_set<std::shared_ptr<Value>> values;
};

#endif  // MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
