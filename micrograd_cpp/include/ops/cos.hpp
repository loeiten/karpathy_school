#ifndef MICROGRAD_CPP_INCLUDE_OPS_COS_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_COS_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Cos: private Op{
  public:
    Cos(std::shared_ptr<Value> arg);
    Value &Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> arg_;
    std::shared_ptr<Value> out_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_COS_HPP_
