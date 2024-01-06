#ifndef MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_

#include <memory>  // for shared_ptr

#include "op.hpp"  // for Op

class Value;

class Pow : private Op {
 public:
  Pow(std::shared_ptr<Value> base, const double &exponent);
  Value &Forward() final;
  void Backward() final;

 private:
  std::shared_ptr<Value> base_;
  const double exponent_;
  std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_
