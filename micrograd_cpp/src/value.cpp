#include "../include/value.hpp"

#include <cmath>
#include <iomanip>        // for operator<<, setprecision
#include <unordered_set>  // for unordered_set

int Value::instance_count = 0;

std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value::Value(const double &data, const std::string &label)
    : data_(data), grad_(0), label_(label) {
  ++instance_count;
  id_ = instance_count;
}

Value::Value(const double &data, std::set<Value *> &&children,
             const std::string &op)
    : data_(data), grad_(0), prev_(children), op_(op) {
  ++instance_count;
  id_ = instance_count;
}

Value Value::operator+(Value &rhs) {  // NOLINT
  std::set<Value *> children;
  children.insert(this);
  children.insert(&rhs);
  // We move the children to the out object, no reason for copy as the variable
  // in this function will go out of scope anyways
  Value out = Value(this->data_ + rhs.data_, std::move(children), "+");

  // Finally, we define the backward function
  out.Backward_ = [this, &out, &rhs]() {
    this->grad_ = out.grad_;
    rhs.grad_ = out.grad_;
  };
  return out;
}

Value Value::operator*(Value &rhs) {  // NOLINT
  std::set<Value *> children;
  children.insert(this);
  children.insert(&rhs);
  Value out(this->data_ * rhs.data_, std::move(children), "*");
  out.Backward_ = [this, &out, &rhs]() {
    this->grad_ = rhs.data_ * out.grad_;
    rhs.grad_ = this->data_ * out.grad_;
  };
  return out;
}

const std::set<Value *> &Value::get_children() const { return prev_; }

const std::string &Value::get_op() const { return op_; }

int Value::get_id() const { return id_; }

void Value::set_label(const std::string &label) { label_ = label; }

void Value::set_grad(const double &grad) { grad_ = grad; }

Value Value::tanh() {
  const double &x = data_;
  const double &t = (exp(2 * x) - 1) / (exp(2 * x) + 1);
  std::set<Value *> children;
  children.insert(this);
  Value out(t, std::move(children), "tanh");
  out.Backward_ = [this, &out, &t]() {
    this->grad_ = (1 - pow(t, 2)) * out.grad_;
  };
  return out;
}

void Value::Backward() {
  // Build the topology
  BuildTopo(*this);
  this->grad_ = 1.0;

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->Backward_();
  }
}

void Value::BuildTopo(const Value &value) {
  (void)value;
  if (visited.find(&value) == visited.end()) {
    visited.insert(&value);
    for (const auto &child : value.prev_) {
      BuildTopo(*child);
    }
    topo.push_back(&value);
  }
}
