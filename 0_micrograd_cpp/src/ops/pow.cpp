#include "../../include/ops/pow.hpp"

#include <cmath>    // for pow
#include <iomanip>  // for operator<<, setprecision
#include <memory>   // for allocator, shared_ptr
#include <sstream>  // for char_traits, basic_ostream, oper...

#include "../../include/graph.hpp"   // for Graph
#include "../../include/ops/op.hpp"  // for Op
#include "../../include/value.hpp"   // for Value

Pow::Pow(std::shared_ptr<Value> base, const double &exponent)
    : Op(base), base_(base), exponent_(exponent) {}

Value &Pow::Forward() {
  auto &out = graph.CreateValue(std::pow(base_->get_data(), exponent_));
  out_ = out.get_shared_ptr();
  out.AddProducer(base_);
  std::stringstream ss;
  ss << "^(" << std::fixed << std::setprecision(2) << exponent_ << ")";
  out.set_op(ss.str());
  // Reset the sstream
  ss.str("");
  ss.clear();
  ss << "pow_out_id_" << out.get_id();
  out.set_label(ss.str());
  return out;
}

void Pow::Backward() {
  base_->UpdateGrad(exponent_ * std::pow(base_->get_data(), exponent_ - 1) *
                    out_->get_grad());
}
