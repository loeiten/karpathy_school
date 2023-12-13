#ifndef MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Exp: private Op{
  public:
    Exp(std::shared_ptr<Value> exponent);
    Value &Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> exponent_;
    std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_
