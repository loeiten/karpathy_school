#include "../../include/ops/add.hpp"
#include "../../include/ops/sub.hpp"
#include "../../include/ops/neg.hpp"

#include <memory>
#include <sstream>
#include <iomanip>  // for operator<<, setprecision

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Sub::Sub(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Sub::Sub(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto& tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Sub::Sub(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto& tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Sub::Forward() {
  auto neg_op = Neg(rhs_);
  auto &tmp = neg_op.Forward();
  auto tmpPtr = tmp.get_shared_ptr();
  auto add_op = Add(lhs_, tmpPtr);
  auto &out = add_op.Forward();
  return out;
}

void Sub::Backward() {
  // NOTE: The back-propagation is controlled by the the other backward passes
}