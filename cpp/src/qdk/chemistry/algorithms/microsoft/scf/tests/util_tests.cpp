// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/util/cache.h>
#include <qdk/chemistry/scf/util/class_registry.h>
#include <qdk/chemistry/scf/util/singleton.h>

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

//==============================================================================
// Cache Tests
//==============================================================================

using namespace qdk::chemistry::scf;
struct Dummy {
  int x;
  std::string y;
  Dummy(int x_, std::string y_) : x(x_), y(std::move(y_)) {}
};

TEST(CacheTest, InsertAndRetrieve) {
  util::Cache<Dummy> cache;
  Dummy* obj = cache.emplace(1, 42, "bar");
  ASSERT_NE(obj, nullptr);
  EXPECT_EQ(obj->x, 42);
  EXPECT_EQ(obj->y, "bar");
  // Retrieval should return the same pointer
  Dummy* obj2 = cache.get(1);
  EXPECT_EQ(obj, obj2);
}

TEST(CacheTest, NotFound) {
  util::Cache<Dummy> cache;
  EXPECT_EQ(cache.get(999), nullptr);
}

TEST(CacheTest, EraseAndClear) {
  util::Cache<Dummy> cache;
  cache.emplace(1, 1, "one");
  cache.emplace(2, 2, "two");
  cache.erase(1);
  EXPECT_EQ(cache.get(1), nullptr);
  EXPECT_NE(cache.get(2), nullptr);
  cache.clear();
  EXPECT_EQ(cache.get(2), nullptr);
}

TEST(CacheTest, FindByIdentifierAndOperator) {
  util::Cache<Dummy> cache;
  std::string key = "hello";
  auto idx = std::hash<std::string>{}(key);
  cache.emplace(idx, 123, "world");
  // Test find_by_identifier
  Dummy* found = cache.find_by_identifier(key);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->x, 123);
  EXPECT_EQ(found->y, "world");
  // Test operator[]
  Dummy* found2 = cache[key];
  EXPECT_EQ(found2, found);
  // Test const version
  const util::Cache<Dummy>& ccache = cache;
  const Dummy* cfound = ccache.find_by_identifier(key);
  EXPECT_EQ(cfound, found);
  const Dummy* cfound2 = ccache[key];
  EXPECT_EQ(cfound2, found);
}

//==============================================================================
// Singleton Tests
//==============================================================================

// A simple class for testing the Singleton utility
class TestSingleton {
 public:
  TestSingleton() : value_(0) {}

  explicit TestSingleton(int value) : value_(value) {}

  void setValue(int value) { value_ = value; }

  int getValue() const { return value_; }

  void increment() { ++value_; }

 private:
  int value_;
};

TEST(SingletonTest, BasicFunctionality) {
  using singleton_type = util::Singleton<TestSingleton>;
  // Reset any existing instance
  singleton_type::reset();

  // Get the singleton instance
  TestSingleton& instance1 = singleton_type::instance();

  // Set a value
  instance1.setValue(42);

  // Get another reference to the singleton instance
  TestSingleton& instance2 = singleton_type::instance();

  // Verify it's the same instance
  EXPECT_EQ(&instance1, &instance2);
  EXPECT_EQ(instance1.getValue(), 42);
  EXPECT_EQ(instance2.getValue(), 42);

  // Modify through the second reference
  instance2.setValue(100);

  // Verify the change is visible through the first reference
  EXPECT_EQ(instance1.getValue(), 100);
}

TEST(SingletonTest, ConstructorArgs) {
  using singleton_type = util::Singleton<TestSingleton>;
  // Reset any existing instance
  singleton_type::reset();

  // Get the singleton instance with constructor arguments
  TestSingleton& instance = singleton_type::instance(999);

  // Verify the constructor argument was used
  EXPECT_EQ(instance.getValue(), 999);
}

TEST(SingletonTest, MultipleTypes) {
  // Define a second test class
  class AnotherTestSingleton {
   public:
    AnotherTestSingleton() : value_("default") {}
    explicit AnotherTestSingleton(std::string value)
        : value_(std::move(value)) {}

    const std::string& getValue() const { return value_; }
    void setValue(const std::string& value) { value_ = value; }

   private:
    std::string value_;
  };

  using singleton_type = util::Singleton<TestSingleton>;
  using another_singleton_type = util::Singleton<AnotherTestSingleton>;

  // Reset any existing instances
  singleton_type::reset();
  another_singleton_type::reset();

  // Get instance of the first singleton type
  TestSingleton& instance1 = singleton_type::instance();
  instance1.setValue(42);

  // Get instance of the second singleton type
  AnotherTestSingleton& instance2 = another_singleton_type::instance();
  instance2.setValue("test-value");

  // Verify both singletons maintain their separate states
  EXPECT_EQ(instance1.getValue(), 42);
  EXPECT_EQ(instance2.getValue(), "test-value");
}

TEST(SingletonTest, ThreadSafety) {
  constexpr int NUM_THREADS = 10;

  // Use atomic variables and a mutex for thread synchronization in C++17
  std::atomic<int> ready_count(0);
  std::atomic<bool> start_flag(false);
  std::mutex mutex;
  std::condition_variable cv;

  // Use a unique class for this test to ensure we start with a fresh singleton
  class ThreadTestSingleton {
   public:
    ThreadTestSingleton() : counter_(0) {}
    void increment() { ++counter_; }
    int getCounter() const { return counter_; }

   private:
    std::atomic<int> counter_;
  };

  // Reset singleton at the start of the test
  util::Singleton<ThreadTestSingleton>::reset();

  auto thread_func = [&]() {
    // Signal that this thread is ready
    ready_count++;

    // Wait for all threads to be ready
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() { return start_flag.load(); });
    }

    // Get the singleton instance and increment the counter
    auto& singleton = util::Singleton<ThreadTestSingleton>::instance();
    for (int i = 0; i < 1000; ++i) {
      singleton.increment();
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back(thread_func);
  }

  // Wait for all threads to be ready
  while (ready_count.load() < NUM_THREADS) {
    std::this_thread::yield();
  }

  // Signal all threads to start
  {
    std::lock_guard<std::mutex> lock(mutex);
    start_flag = true;
  }
  cv.notify_all();

  // Wait for all threads to complete
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Verify the counter value
  auto& singleton = util::Singleton<ThreadTestSingleton>::instance();
  EXPECT_EQ(singleton.getCounter(), NUM_THREADS * 1000);
}

//==============================================================================
// ClassRegistry Tests
//==============================================================================

// A simple class for testing the ClassRegistry
class TestObject {
 public:
  TestObject() : value_(0) {}
  explicit TestObject(int value) : value_(value) {}

  void setValue(int value) { value_ = value; }

  int getValue() const { return value_; }

 private:
  int value_;
};

// Test identifier class with custom hash implementation
class TestIdentifier {
 public:
  explicit TestIdentifier(int id) : id_(id) {}

  int getId() const { return id_; }

  bool operator==(const TestIdentifier& other) const {
    return id_ == other.id_;
  }

 private:
  int id_;
};

// Hash specialization for TestIdentifier
namespace std {
template <>
struct hash<TestIdentifier> {
  size_t operator()(const TestIdentifier& id) const {
    return std::hash<int>{}(id.getId());
  }
};
}  // namespace std

// Specialized ClassRegistry for TestObject
using TestRegistry = util::ClassRegistry<TestObject, TestIdentifier>;

TEST(ClassRegistryTest, BasicFunctionality) {
  // Clear any existing entries in case other tests populated the registry
  TestRegistry::clear();

  // Create an identifier
  TestIdentifier id1(1);

  // Initially, the registry should not contain the object
  EXPECT_EQ(TestRegistry::find(id1), nullptr);

  // Create and store an object
  TestObject* obj1 = TestRegistry::get_or_create(id1, 42);
  EXPECT_NE(obj1, nullptr);
  EXPECT_EQ(obj1->getValue(), 42);

  // Verify the object can be retrieved from the registry
  TestObject* found1 = TestRegistry::find(id1);
  EXPECT_EQ(found1, obj1);
  EXPECT_EQ(found1->getValue(), 42);

  // Modify the object
  obj1->setValue(100);

  // Verify the change is visible through the registry
  TestObject* found2 = TestRegistry::find(id1);
  EXPECT_EQ(found2, obj1);
  EXPECT_EQ(found2->getValue(), 100);

  // Remove the object from the registry
  TestRegistry::remove(id1);

  // Verify the object is no longer in the registry
  EXPECT_EQ(TestRegistry::find(id1), nullptr);
}

TEST(ClassRegistryTest, MultipleObjects) {
  // Clear any existing entries
  TestRegistry::clear();

  // Create multiple identifiers
  TestIdentifier id1(1);
  TestIdentifier id2(2);
  TestIdentifier id3(3);

  // Create and store multiple objects
  TestObject* obj1 = TestRegistry::get_or_create(id1, 10);
  TestObject* obj2 = TestRegistry::get_or_create(id2, 20);
  TestObject* obj3 = TestRegistry::get_or_create(id3, 30);

  // Verify all objects were created with correct values
  EXPECT_EQ(obj1->getValue(), 10);
  EXPECT_EQ(obj2->getValue(), 20);
  EXPECT_EQ(obj3->getValue(), 30);

  // Verify objects can be retrieved independently
  EXPECT_EQ(TestRegistry::find(id1), obj1);
  EXPECT_EQ(TestRegistry::find(id2), obj2);
  EXPECT_EQ(TestRegistry::find(id3), obj3);

  // Remove one object
  TestRegistry::remove(id2);

  // Verify only that object is removed
  EXPECT_NE(TestRegistry::find(id1), nullptr);
  EXPECT_EQ(TestRegistry::find(id2), nullptr);
  EXPECT_NE(TestRegistry::find(id3), nullptr);

  // Clear all objects
  TestRegistry::clear();

  // Verify all objects are removed
  EXPECT_EQ(TestRegistry::find(id1), nullptr);
  EXPECT_EQ(TestRegistry::find(id2), nullptr);
  EXPECT_EQ(TestRegistry::find(id3), nullptr);
}

TEST(ClassRegistryTest, GetOrCreateReuse) {
  // Clear any existing entries
  TestRegistry::clear();

  // Create an identifier
  TestIdentifier id(1);

  // Create an object
  TestObject* obj1 = TestRegistry::get_or_create(id, 42);

  // Modify the object
  obj1->setValue(100);

  // Attempt to create an object with the same id but different constructor
  // value
  TestObject* obj2 = TestRegistry::get_or_create(id, 200);

  // Verify the existing object was returned (not a new one)
  EXPECT_EQ(obj1, obj2);
  EXPECT_EQ(obj2->getValue(),
            100);  // Should still have the modified value, not 200
}

TEST(ClassRegistryTest, MultipleRegistries) {
  // Define a second test class
  class AnotherTestObject {
   public:
    AnotherTestObject() : name_("default") {}
    explicit AnotherTestObject(std::string name) : name_(std::move(name)) {}

    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

   private:
    std::string name_;
  };

  // Create another registry type
  using AnotherRegistry =
      util::ClassRegistry<AnotherTestObject, TestIdentifier>;

  // Clear both registries
  TestRegistry::clear();
  AnotherRegistry::clear();

  // Create an identifier
  TestIdentifier id(1);

  // Create objects in both registries
  TestObject* obj1 = TestRegistry::get_or_create(id, 42);
  AnotherTestObject* obj2 = AnotherRegistry::get_or_create(id, "test-name");

  // Verify both objects were created
  EXPECT_NE(obj1, nullptr);
  EXPECT_NE(obj2, nullptr);

  // Verify both objects have their own values
  EXPECT_EQ(obj1->getValue(), 42);
  EXPECT_EQ(obj2->getName(), "test-name");

  // Modify the objects
  obj1->setValue(100);
  obj2->setName("updated-name");

  // Verify the changes are visible through the registries
  TestObject* found1 = TestRegistry::find(id);
  AnotherTestObject* found2 = AnotherRegistry::find(id);

  EXPECT_EQ(found1->getValue(), 100);
  EXPECT_EQ(found2->getName(), "updated-name");
}

// Test with primitive type as key
TEST(ClassRegistryTest, PrimitiveKey) {
  // Create a registry with int as the key type
  using IntKeyRegistry = util::ClassRegistry<TestObject, int>;

  // Clear the registry
  IntKeyRegistry::clear();

  // Create and store objects with int keys
  TestObject* obj1 = IntKeyRegistry::get_or_create(42, 100);
  TestObject* obj2 = IntKeyRegistry::get_or_create(99, 200);

  // Verify objects were created with correct values
  EXPECT_EQ(obj1->getValue(), 100);
  EXPECT_EQ(obj2->getValue(), 200);

  // Verify objects can be retrieved by integer keys
  EXPECT_EQ(IntKeyRegistry::find(42), obj1);
  EXPECT_EQ(IntKeyRegistry::find(99), obj2);
  EXPECT_EQ(IntKeyRegistry::find(100), nullptr);  // Non-existent key
}
