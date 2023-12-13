#ifndef MICROGRAD_CPP_INCLUDE_OPS_SUB_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_SUB_HPP_

#include <memory>

#include "op.hpp"

class Value;

class Sub : private Op {
public:
  Sub(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  Sub(std::shared_ptr<Value> lhs, const double &rhs);
  Sub(const double &lhs, std::shared_ptr<Value> rhs);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> rhs_;
  std::shared_ptr<Value> lhs_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_SUB_HPP_
