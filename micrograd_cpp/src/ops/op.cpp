#include "../../include/ops/op.hpp"
#include "../../include/value.hpp"
#include <memory>

Op::Op(std::shared_ptr<Value> val) : graph(val->get_graph()) {}
