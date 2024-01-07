#ifndef MICROGRAD_CPP_INCLUDE_OPS_TANH_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_TANH_HPP_

#include <memory>  // for shared_ptr

#include "../../include/ops/op.hpp"  // for Op

class Value;

class Tanh : private Op {
 public:
  explicit Tanh(std::shared_ptr<Value> arg);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> arg_;
  double t_ = 0.0;  // Helper which avoids recomputation
  std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_TANH_HPP_
