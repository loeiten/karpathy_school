#ifndef MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_

#include <memory>         // for shared_ptr
#include <string>         // for string
#include <unordered_set>  // for unordered_set
#include <utility>        // for pair

class Value;

void Trace(
    const Value &value, std::unordered_set<std::shared_ptr<Value>> *nodes,
    std::unordered_set<
        std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>> *edges);

std::string ReturnDot(const Value &root);

#endif  // MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
