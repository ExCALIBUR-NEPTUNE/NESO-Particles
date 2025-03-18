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

struct ResourceStackInterfaceDouble : ResourceStackInterface<double> {

  int count{0};
  double reset;

  ResourceStackInterfaceDouble(double reset) { this->reset = reset; }

  virtual inline std::shared_ptr<double> construct() override {
    this->count++;
    auto i = std::make_shared<double>();
    *i = reset;
    return i;
  }

  virtual inline void free(std::shared_ptr<double> &resource) override {
    this->count--;
    *resource = -1.1;
  }

  virtual inline void clean(std::shared_ptr<double> &resource) override {
    *resource = this->reset;
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

TEST(ResourceStack, resource_stack_map) {
  auto resource_stack_map = std::make_shared<ResourceStackMap>();

  struct A {};
  struct B {};

  EXPECT_FALSE(resource_stack_map->exists(A{}));
  EXPECT_FALSE(resource_stack_map->exists(B{}));

  auto A_stack = create_resource_stack<int, ResourceStackInterfaceInt>();
  resource_stack_map->set(A{}, A_stack);

  EXPECT_TRUE(resource_stack_map->exists(A{}));
  EXPECT_EQ(resource_stack_map->get<ResourceStack<int>>(A{}).get(),
            A_stack.get());
  EXPECT_FALSE(resource_stack_map->exists(B{}));

  auto B_stack =
      create_resource_stack<double, ResourceStackInterfaceDouble>(3.14);
  resource_stack_map->set(B{}, B_stack);

  EXPECT_TRUE(resource_stack_map->exists(A{}));
  EXPECT_TRUE(resource_stack_map->exists(B{}));
  EXPECT_EQ(resource_stack_map->get<ResourceStack<int>>(A{}).get(),
            A_stack.get());
  EXPECT_EQ(resource_stack_map->get<ResourceStack<double>>(B{}).get(),
            B_stack.get());

  auto d0 = resource_stack_map->get<ResourceStack<double>>(B{})->get();
  EXPECT_EQ(*d0, 3.14);
  resource_stack_map->get<ResourceStack<double>>(B{})->restore(d0);

  auto B_stack2 = get_resource_stack<double, ResourceStackInterfaceDouble>(
      resource_stack_map, B{}, 3.14);

  EXPECT_EQ(B_stack2.get(), B_stack.get());

  d0 = get_resource<double, ResourceStackInterfaceDouble>(resource_stack_map,
                                                          B{}, 3.14);
  EXPECT_EQ(*d0, 3.14);
  restore_resource(resource_stack_map, B{}, d0);

  resource_stack_map->free();
}

TEST(ResourceStack, resource_stack_buffer_device_host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  {
    auto tmp_buffer =
        get_resource<BufferDeviceHost<REAL>,
                     ResourceStackInterfaceBufferDeviceHost<REAL>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDeviceHost<REAL>{}, sycl_target);
    tmp_buffer->realloc_no_copy(128);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<REAL>{}, tmp_buffer);
  }

  {
    auto tmp_buffer =
        get_resource<BufferDeviceHost<REAL>,
                     ResourceStackInterfaceBufferDeviceHost<REAL>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDeviceHost<REAL>{}, sycl_target);
    EXPECT_EQ(tmp_buffer->size, 128);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<REAL>{}, tmp_buffer);
  }
  sycl_target->free();
}
