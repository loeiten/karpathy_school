#include "../../include/ops/neg.hpp"
#include "../../include/ops/mul.hpp"

#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Neg::Neg(std::shared_ptr<Value> val) : Op(val), val_(val) {}

Value &Neg::Forward() {
  auto mul_op = std::make_shared<Mul>(val_, -1.0);
  auto &out = mul_op->Forward();
  out.set_backward(std::bind(&Mul::Backward, mul_op));
  out.AddProducer(val_);
  out_ = out.get_shared_ptr();
  std::stringstream ss;
  ss << "neg_out_id_" << val_->get_id();
  out_->set_label(ss.str());
  return out;
}

void Neg::Backward() {
  out_->Backward();
}