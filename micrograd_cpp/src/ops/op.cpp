#include "../../include/ops/op.hpp"

#include <memory>  // for shared_ptr

#include "../../include/value.hpp"

Op::Op(std::shared_ptr<Value> val) : graph(val->get_graph()) {}
