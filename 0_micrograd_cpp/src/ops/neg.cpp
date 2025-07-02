#include "../../include/ops/neg.hpp"

#include <functional>  // for __bind, bind
#include <memory>      // for shared_ptr, make_shared
#include <sstream>     // for char_traits, basic_ostream, ope...

#include "../../include/ops/mul.hpp"  // for Mul
#include "../../include/ops/op.hpp"   // for Op
#include "../../include/value.hpp"    // for Value

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

void Neg::Backward() { out_->Backward(); }
