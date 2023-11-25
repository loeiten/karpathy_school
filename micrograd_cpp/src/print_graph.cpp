#include <sstream>
#include <unordered_set>
#include <utility>

#include "../include/value.hpp"

void Trace(const Value &value, std::unordered_set<Value> *nodes,
           std::unordered_set<std::pair<Value, Value>> *edges) {
  if (nodes->find(value) == nodes->end()) {
    nodes->emplace(value);
    for (const auto &child : value.get_children()) {
      edges->emplace(std::make_pair(*child, value));
      Trace(*child, nodes, edges);
    }
  }
}

std::string ReturnDot(const Value &root) {
  const int indent = 4;
  // FIXME: Here the Values are "hard copies", would it make sense to convert
  //        them to pointers?
  std::unordered_set<Value> nodes;
  std::unordered_set<std::pair<Value, Value>> edges;
  Trace(root, &nodes, &edges);

  std::stringstream dot_stream;
  dot_stream << "digraph {\n"
             << std::string(indent, ' ') << "graph [rankdir=LR]\n";

  // Create the nodes and connect the op to it's children
  for (const auto &node : nodes) {
    // Create the node
    dot_stream << std::string(indent, ' ') << "\"" << node.get_id() << "\""
               << " [label=\"{" << node << "}\" shape=record]\n";
    if (node.get_op() != "") {
      // Connect op to the children
      dot_stream << std::string(indent, ' ') << "\"" << node.get_id()
                 << node.get_op() << "\" [label=\"" << node.get_op() << "\"]\n";
      dot_stream << std::string(indent, ' ') << "\"" << node.get_id()
                 << node.get_op() << "\" -> \"" << node.get_id() << "\""
                 << "\n";
    }
  }

  // Create the rest of the edges
  for (const auto &child_parent : edges) {
    auto &child = child_parent.first;
    auto &parent = child_parent.second;
    dot_stream << std::string(indent, ' ') << "\"" << child.get_id()
               << "\" -> \"" << parent.get_id() << parent.get_op() << "\"\n";
  }

  dot_stream << "}\n";
  return dot_stream.str();
}

/*
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #,
node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data,
n.grad), shape='record') if n._op: dot.node(name=str(id(n)) + n._op,
label=n._op) dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

digraph {
        graph [rankdir=LR]
        4414566560 [label="{ data 2.0000 | grad 1.0000 }" shape=record]
        "4414566560*" [label="*"]
        "4414566560*" -> 4414566560
        4414445264 [label="{ data 3.0000 | grad 1.0000 }" shape=record]
        "4414445264ReLU" [label="ReLU"]
        "4414445264ReLU" -> 4414445264
        4414569728 [label="{ data 2.0000 | grad 1.0000 }" shape=record]
        4414408976 [label="{ data 1.0000 | grad 1.0000 }" shape=record]
        4414410176 [label="{ data 3.0000 | grad 1.0000 }" shape=record]
        "4414410176+" [label="+"]
        "4414410176+" -> 4414410176
        4414568432 [label="{ data 1.0000 | grad 2.0000 }" shape=record]
        4414566560 -> "4414410176+"
        4414569728 -> "4414566560*"
        4414410176 -> "4414445264ReLU"
        4414568432 -> "4414566560*"
        4414408976 -> "4414410176+"
}
*/
