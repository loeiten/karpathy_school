#include "../../include/ops/neg.hpp"
#include "../../include/ops/mul.hpp"

#include <memory>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Neg::Neg(std::shared_ptr<Value> val) : Op(val), val_(val) {}

Value &Neg::Forward() {
  auto mul_op = Mul(val_, -1.0);
  auto &out = mul_op.Forward();
  return out;
}

void Neg::Backward() {
  // NOTE: The back-propagation is controlled by the the other backward passes
}