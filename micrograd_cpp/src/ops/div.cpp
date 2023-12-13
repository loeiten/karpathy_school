#include "../../include/ops/div.hpp"

#include <memory>
#include <sstream>
#include <iomanip>  // for operator<<, setprecision

#include "../../include/graph.hpp"
#include "../../include/ops/mul.hpp"
#include "../../include/ops/pow.hpp"
#include "../../include/value.hpp"

Div::Div(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}
Div::Div(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto& tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}
Div::Div(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto& tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Div::Forward() {
  auto pow_op = Pow(rhs_, -1.0);
  auto &tmp = pow_op.Forward();
  auto tmpPtr = tmp.get_shared_ptr();
  auto mul_op = Mul(lhs_, tmpPtr);
  auto &out = mul_op.Forward();
  return out;
}

void Div::Backward() {
  // NOTE: The back-propagation is controlled by the the other backward passes
}