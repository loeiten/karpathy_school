#ifndef MICROGRAD_CPP_INCLUDE_OP_HPP_
#define MICROGRAD_CPP_INCLUDE_OP_HPP_

class Op{
    virtual void Forward() = 0;
    virtual void Backward() = 0;
};

/*
I have a graph like so

t1        t4
  \        \
    + - t3 * - t5 - cos - t6
  /
t2 

I want the user to be able to write something like this
((t1 + t2) * t4).cos()

hence
t3 = t1 + t2
t5 = t3 * t4

If we ask how much t1 contribute to t6, we get

d/dt1 t6 = d/dt1 t6( t5(t4, t3(t2, t1)) )
         = dt6/dt6 * dt6/dt5 * dt5/dt3 * dt3/dt1

This means that the gradient of t1 will be dependent on t3, said
differently t3 will modify t1

Questions:
- Where should the ownership be of the t nodes
  - Graph should own it, but how will I be able to do chaining then?
    Comment: One could use dereference and change to
    (mul(add(*t1, *t2), *t4))->cos()
    (*(*t1 + *t2) * *t4)->cos()

Pattern called: PIMPL (pointer to implementation)
ValueImpl  - class having all the actual data
Value - shell containing overloaded ops, and a weak_ptr to ValueImpl
Graph - owns the ValueImpl and stores it as a shared_ptr in some container
End user should not have access to ValueImpl
Op - class which has fwd and bwd and all inputs and outputs
*/


#endif  // MICROGRAD_CPP_INCLUDE_OP_HPP_
