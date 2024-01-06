#ifndef MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_

#include <memory>  // for shared_ptr

#include "op.hpp"  // for Op

class Value;

class Add : private Op {
 public:
  Add(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  Add(std::shared_ptr<Value> lhs, const double &rhs);
  Add(const double &lhs, std::shared_ptr<Value> rhs);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> rhs_;
  std::shared_ptr<Value> lhs_;
  std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_
