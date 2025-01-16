#include "include/test_neso_particles.hpp"

namespace {

struct ResourceStackInterfaceInt : ResourceStackInterface<int> {

  int count{0};

  virtual inline std::shared_ptr<int> construct() override {
    this->count++;
    auto i = std::make_shared<int>();
    *i = 42;
    return i;
  }

  virtual inline void free(std::shared_ptr<int> &resource) override {
    this->count--;
    *resource = -1;
  }

  virtual inline void clean(std::shared_ptr<int> &resource) override {
    *resource = 42;
  }
};

} // namespace

TEST(ResourceStack, base) {

  auto resource_stack_interface_int =
      std::make_shared<ResourceStackInterfaceInt>();
  ASSERT_EQ(resource_stack_interface_int->count, 0);
  auto resource_stack =
      std::make_shared<ResourceStack<int>>(resource_stack_interface_int);

  auto i0 = resource_stack->get();
  ASSERT_EQ(*i0, 42);
  ASSERT_EQ(resource_stack_interface_int->count, 1);
  ASSERT_EQ(resource_stack->stack.size(), 0);

  *i0 = 43;
  resource_stack->restore(i0);
  ASSERT_EQ(resource_stack_interface_int->count, 1);
  ASSERT_EQ(resource_stack->stack.size(), 1);
  ASSERT_EQ(i0.get(), nullptr);

  i0 = resource_stack->get();
  ASSERT_EQ(*i0, 42);
  *i0 = 43;
  ASSERT_EQ(resource_stack_interface_int->count, 1);
  ASSERT_EQ(resource_stack->stack.size(), 0);
  auto i1 = resource_stack->get();
  ASSERT_EQ(resource_stack_interface_int->count, 2);
  ASSERT_EQ(resource_stack->stack.size(), 0);
  ASSERT_EQ(*i1, 42);
  *i1 = 43;

  resource_stack->restore(i0);
  resource_stack->restore(i1);
  ASSERT_EQ(i0.get(), nullptr);
  ASSERT_EQ(i1.get(), nullptr);

  ASSERT_EQ(resource_stack->stack.size(), 2);

  resource_stack->free();
  ASSERT_EQ(resource_stack->stack.size(), 0);
  ASSERT_EQ(resource_stack_interface_int->count, 0);
}
