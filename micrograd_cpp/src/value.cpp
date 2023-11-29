#include "../include/value.hpp"

#include <iomanip>        // for operator<<, setprecision
#include <memory>         // for shared_ptr, make_shared
#include <unordered_set>  // for unordered_set

int Value::instance_count = 0;

std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value::Value(const float &data, const std::string &label)
    : data_(data), grad_(0), label_(label) {
  ++instance_count;
  id_ = instance_count;
}

Value::Value(const float &data,
             std::unordered_set<std::shared_ptr<Value>> &&children,
             const std::string &op)
    : data_(data), grad_(0), prev_(children), op_(op) {
  ++instance_count;
  id_ = instance_count;
}

Value Value::operator+(const Value &rhs) const {
  std::unordered_set<std::shared_ptr<Value>> children;
  // Emplace will create the object in place in order to save a copy (first
  // create, then copy it to the unordered set)
  children.emplace(std::make_shared<Value>(*this));
  children.emplace(std::make_shared<Value>(rhs));
  // We move the children to the out object, no reason for copy as the variable
  // in this function will go out of scope anyways
  Value out = Value(this->data_ + rhs.data_, std::move(children), "+");
  return out;
}

Value Value::operator*(const Value &rhs) const {
  std::unordered_set<std::shared_ptr<Value>> children;
  children.emplace(std::make_shared<Value>(*this));
  children.emplace(std::make_shared<Value>(rhs));
  Value out(this->data_ * rhs.data_, std::move(children), "*");
  return out;
}

const std::unordered_set<std::shared_ptr<Value>> &Value::get_children() const {
  return prev_;
}

const std::string &Value::get_op() const { return op_; }

int Value::get_id() const { return id_; }

void Value::set_label(const std::string &label) { label_ = label; }
