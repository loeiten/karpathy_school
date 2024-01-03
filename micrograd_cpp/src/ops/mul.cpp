#include "../../include/ops/mul.hpp"

#include <memory>
#include <sstream>
#include <iomanip>  // for operator<<, setprecision

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Mul::Mul(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Mul::Mul(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto& tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Mul::Mul(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto& tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Mul::Forward() {
  // FIXME:
  std::cout << "Mul FWD:" << std::endl;
  std::cout << "  rhs_: id " << rhs_->get_id() << " | " << *rhs_ << std::endl;
  std::cout << "  lhs_: id " << lhs_->get_id() << " | " << *lhs_ << std::endl;
  auto &out = graph.CreateValue(lhs_->get_data() * rhs_->get_data());
  out_ = out.get_shared_ptr();
  out.AddProducer(lhs_);
  out.AddProducer(rhs_);
  out.set_op("*");
  return out;
  // FIXME:
  std::cout << "  out: id " << out.get_id() << " | " << out << std::endl;
}

void Mul::Backward() {
  // FIXME:
  std::cout << "        Mul BWD:" << std::endl;
  std::cout << "          rhs_: id " << rhs_->get_id() << " | " << *rhs_ << std::endl;
  std::cout << "          lhs_: id " << lhs_->get_id() << " | " << *lhs_ << std::endl;
  std::cout << "          out_: id " << out_->get_id() << " | " << *out_ << std::endl;
  lhs_->UpdateGrad(rhs_->get_data() * out_->get_grad());
  rhs_->UpdateGrad(lhs_->get_data() * out_->get_grad());
}