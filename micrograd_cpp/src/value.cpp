#include "../include/value.hpp"

#include <iomanip>
#include <memory>
#include <unordered_set>

int Value::instance_count = 0;

std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.data_;
  return os;
}

Value::Value(const float &data) : data_(data) {
  ++instance_count;
  id_ = instance_count;
};

Value::Value(const float &data,
             std::unordered_set<std::shared_ptr<Value>> &&children,
             const std::string &op)
    : data_(data), prev_(children), op_(op) {
  ++instance_count;
  id_ = instance_count;
};

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

std::unordered_set<std::shared_ptr<Value>> Value::get_children() const {
  return prev_;
}

std::string Value::get_op() const { return op_; }

int Value::get_id() const { return id_; }
