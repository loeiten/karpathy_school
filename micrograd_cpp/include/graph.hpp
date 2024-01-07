#ifndef MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_GRAPH_HPP_

#include <memory>         // for shared_ptr
#include <set>            // for set
#include <string>         // for string
#include <unordered_set>  // for unordered_set
#include <utility>        // for pair
#include <vector>         // for vector

class Value;

class Graph {
 public:
  /*
  With the following graph

  t1        t4
    \        \
      + - t3 * - t5 - cos - t6
    /
  t2

  We would like the user to be able to write
  cos((t1 + t2) * t4)

  where
  t3 = t1 + t2
  t5 = t3 * t4

  If we ask how much t1 contribute to t6, we get

  d/dt1 t6 = d/dt1 t6( t5(t4, t3(t2, t1)) )
           = dt6/dt6 * dt6/dt5 * dt5/dt3 * dt3/dt1

  This means that the gradient of t1 will be dependent on t3, said
  differently t3 will modify t1

  In order to achieve this the ownership of the t nodes should lay outside
  the Values. A good place to place them is in a graph.
  */
  // NOTE: We are not using references here in order to get references to
  // dangling temporary. For details see:
  // https://comp.lang.cpp.moderated.narkive.com/FScmAZiw/dangling-reference
  Value &CreateValue(const double value);
  Value &CreateValue(const double value, const std::string &label);

  void TopologicalSort(const Value &value);

  // NOTE: We use set here as we want the printed result to be reproducible
  void Trace(const Value &value, std::set<const std::shared_ptr<Value>> *nodes,
             std::set<std::pair<const std::shared_ptr<Value>,
                                const std::shared_ptr<Value>>> *edges);

  std::string ReturnDot(const Value &root);
  std::string ReturnDot(const std::vector<std::shared_ptr<Value>> &x);

  std::vector<const std::shared_ptr<Value>> topology;

 private:
  std::unordered_set<std::shared_ptr<Value>> values;
  std::unordered_set<int> visited;

  void RecursiveTopologicalSort(const Value &value);

  std::string CreateDotString(
      const std::set<const std::shared_ptr<Value>> &nodes,
      const std::set<std::pair<const std::shared_ptr<Value>,
                               const std::shared_ptr<Value>>> &edges);
};

#endif  // MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
