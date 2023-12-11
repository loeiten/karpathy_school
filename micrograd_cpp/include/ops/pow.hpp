#ifndef MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Pow: private Op{
    Pow(std::shared_ptr<Value> base, float &exponent);
    void Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> base_;
    std::shared_ptr<Value> exponent_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_POW_HPP_
