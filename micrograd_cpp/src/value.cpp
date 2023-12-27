#include "../include/value.hpp"

#include <memory.h>

#include <cmath>
#include <iomanip>  // for operator<<, setprecision
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>  // for unordered_set

#include "../include/ops/add.hpp"
#include "../include/ops/cos.hpp"
#include "../include/ops/div.hpp"
#include "../include/ops/exp.hpp"
#include "../include/ops/mul.hpp"
#include "../include/ops/neg.hpp"
#include "../include/ops/pow.hpp"
#include "../include/ops/sub.hpp"
#include "../include/ops/tanh.hpp"

int Value::instance_count = 0;

// Friend functions
// =============================================================================
std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value &pow(Value &a, const double &n) {
  auto pow_op = Pow(a.get_shared_ptr(), n);
  auto &out = pow_op.Forward();
  return out;
}

Value &operator+(const double &lhs, Value &rhs) {
  auto add_op = Add(lhs, rhs.get_shared_ptr());
  auto &out = add_op.Forward();
  return out;
}

Value &operator+(Value &lhs, const double &rhs) {
  auto add_op = Add(lhs.get_shared_ptr(), rhs);
  auto &out = add_op.Forward();
  return out;
}

Value &operator-(const double &lhs, Value &rhs) {
  auto sub_op = Sub(lhs, rhs.get_shared_ptr());
  auto &out = sub_op.Forward();
  return out;
}

Value &operator-(Value &lhs, const double &rhs) {
  auto sub_op = Sub(lhs.get_shared_ptr(), rhs);
  auto &out = sub_op.Forward();
  return out;
}

Value &operator*(const double &lhs, Value &rhs) {
  auto mul_op = Mul(lhs, rhs.get_shared_ptr());
  auto &out = mul_op.Forward();
  return out;
}

Value &operator*(Value &lhs, const double &rhs) {
  auto mul_op = Mul(lhs.get_shared_ptr(), rhs);
  auto &out = mul_op.Forward();
  return out;
}

Value &operator/(const double &lhs, Value &rhs) {
  auto div_op = Div(lhs, rhs.get_shared_ptr());
  auto &out = div_op.Forward();
  return out;
}

Value &operator/(Value &lhs, const double &rhs) {
  auto div_op = Div(lhs.get_shared_ptr(), rhs);
  auto &out = div_op.Forward();
  return out;
}

Value &tanh(Value &value) {
  auto tanh_op = Tanh(value.get_shared_ptr());
  auto &out = tanh_op.Forward();
  return out;
}

Value &cos(Value &value) {
  auto cos_op = Cos(value.get_shared_ptr());
  auto &out = cos_op.Forward();
  return out;
}

Value &exp(Value &value) {
  auto exp_op = Exp(value.get_shared_ptr());
  auto &out = exp_op.Forward();
  return out;
}
// =============================================================================

// Constructors
// =============================================================================
Value::Value(const double &data) : data_(data), grad_(0) {
  ++instance_count;
  id_ = instance_count;
}

Value::Value(const double &data, const std::string &label)
    : data_(data), grad_(0), label_(label) {
  ++instance_count;
  id_ = instance_count;
}

// FIXME: Check if this is still in use
Value::Value(const double &data, std::set<std::shared_ptr<Value>> &&producers,
             const std::string &op)
    : data_(data), grad_(0), producers(producers), op_(op) {
  ++instance_count;
  id_ = instance_count;
  label_ = "tmp" + std::to_string(id_);
}
// =============================================================================

// Member functions: Operator overloads
// =============================================================================
Value &Value::operator+(Value &rhs) {
  auto add_op = Add(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = add_op.Forward();
  return out;
}

Value &Value::operator*(Value &rhs) {
  auto mul_op = Mul(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = mul_op.Forward();
  return out;
}

Value &Value::operator/(Value &rhs) {
  auto div_op = Div(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = div_op.Forward();
  return out;
}

Value &Value::operator-() {
  auto neg_op = Neg(get_shared_ptr());
  auto &out = neg_op.Forward();
  return out;
}
// =============================================================================

// Member functions: Accessors and mutators
// =============================================================================
const std::set<std::shared_ptr<Value>> &Value::get_producers() const {
  return producers;
}

const std::string &Value::get_op() const { return op_; }

int Value::get_id() const { return id_; }

void Value::set_label(const std::string &label) { label_ = label; }

void Value::set_grad(const double &grad) { grad_ = grad; }
// =============================================================================

// Member functions: Other
// =============================================================================
void Value::AddProducer(std::shared_ptr<Value> producer) {
  producers.insert(producer);
}

void Value::UpdateGrad(const double &grad) { grad_ = grad; }
// =============================================================================

// FIXME: Move to graph
void Value::Backward() {
  // Build the topology
  TopologicalSort(*this);
  this->grad_ = 1.0;

  for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
    if ((*it)->Backward_ != nullptr) {
      (*it)->Backward_();
    }
  }
}

void Value::TopologicalSort(const Value &value) {
  if (visited.find(value.get_id()) == visited.end()) {
    visited.insert(value.get_id());
    for (const auto &child : value.producers) {
      TopologicalSort(*child);
    }
    topology.push_back(value.get_shared_ptr());
  }
}
