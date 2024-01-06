#ifndef MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_

#include <memory>  // for shared_ptr

#include "op.hpp"  // for Op

class Value;

class Neg : private Op {
 public:
  explicit Neg(std::shared_ptr<Value> val);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> val_;
  std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_
