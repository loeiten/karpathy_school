#include "../../include/ops/add.hpp"

#include <iomanip>  // for operator<<, setprecision
#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/ops/op.hpp"
#include "../../include/value.hpp"

Add::Add(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Add::Add(const double& rhs, std::shared_ptr<Value> lhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto& tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Add::Add(std::shared_ptr<Value> rhs, const double& lhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto& tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value& Add::Forward() {
  auto& out = graph.CreateValue(lhs_->get_data() + rhs_->get_data());
  out_ = out.get_shared_ptr();
  out.AddProducer(lhs_);
  out.AddProducer(rhs_);
  out.set_op("+");
  std::stringstream ss;
  ss << "add_out_id_" << out.get_id();
  out.set_label(ss.str());
  return out;
}

void Add::Backward() {
  lhs_->UpdateGrad(out_->get_grad());
  rhs_->UpdateGrad(out_->get_grad());
}
