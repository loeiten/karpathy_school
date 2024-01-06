#ifndef MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_

#include <memory>  // for shared_ptr

#include "op.hpp"  // for Op

class Value;

class Div : private Op {
 public:
  Div(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  Div(std::shared_ptr<Value> lhs, const double &rhs);
  Div(const double &lhs, std::shared_ptr<Value> rhs);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> rhs_;
  std::shared_ptr<Value> lhs_;
  std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_
