#ifndef MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_GRAPH_HPP_

#include <memory>
#include <set>      // for set
#include <string>   // for string
#include <utility>  // for pair

class Value;

class Graph{
    public:
    std::shared_ptr<Value> CreateValue(const std::shared_ptr<Value>& value);
    std::shared_ptr<Value> CreateValue(const float& value);
    // FIXME: Fix these straydogs
    void Trace(const Value &value, std::set<const Value *> *nodes,
               std::set<std::pair<const Value *, const Value *>> *edges);

    std::string ReturnDot(const Value &root);

    private:
    std::set<std::shared_ptr<Value>> values;
};

#endif  // MICROGRAD_CPP_INCLUDE_GRAPH_HPP_
