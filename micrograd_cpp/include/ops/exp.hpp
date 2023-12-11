#ifndef MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Exp: private Op{
    Exp(std::shared_ptr<Value> exponent);
    void Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> exponent_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_EXP_HPP_
