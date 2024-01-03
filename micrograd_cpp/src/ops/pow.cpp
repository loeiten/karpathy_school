#include "../../include/ops/pow.hpp"

#include <iomanip>  // for operator<<, setprecision
#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Pow::Pow(std::shared_ptr<Value> base, const double &exponent)
    : Op(base), base_(base), exponent_(exponent) {}

Value &Pow::Forward() {
  // FIXME:
  std::cout << "Pow FWD:" << std::endl;
  std::cout << "  base_: id " << base_->get_id() << " | " << *base_ << std::endl;
  auto &out = graph.CreateValue(std::pow(base_->get_data(), exponent_));
  out_ = out.get_shared_ptr();
  out.AddProducer(base_);
  std::stringstream ss;
  ss << "^(" << std::fixed << std::setprecision(2) << exponent_ << ")";
  out.set_op(ss.str());
  // FIXME:
  std::cout << "  out: id " << out.get_id() << " | " << out << std::endl;
  return out;
}

void Pow::Backward() {
  std::cout << "        Pow BWD:" << std::endl;
  std::cout << "          base_: id " << base_->get_id() << " | " << *base_ << std::endl;
  std::cout << "          out_: id " << out_->get_id() << " | " << *out_ << std::endl;
  base_->UpdateGrad(exponent_ * std::pow(base_->get_data(), exponent_ - 1) *
                   out_->get_grad());
}