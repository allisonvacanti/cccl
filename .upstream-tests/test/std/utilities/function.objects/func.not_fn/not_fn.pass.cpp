//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// template <class F> unspecified not_fn(F&& f);

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/string>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "type_id.h"


///////////////////////////////////////////////////////////////////////////////
//                       CALLABLE TEST TYPES
///////////////////////////////////////////////////////////////////////////////

bool returns_true() { return true; }

template <class Ret = bool>
struct MoveOnlyCallable {
  MoveOnlyCallable(MoveOnlyCallable const&) = delete;
  MoveOnlyCallable(MoveOnlyCallable&& other)
      : value(other.value)
  { other.value = !other.value; }

  template <class ...Args>
  Ret operator()(Args&&...) { return Ret{value}; }

  explicit MoveOnlyCallable(bool x) : value(x) {}
  Ret value;
};

template <class Ret = bool>
struct CopyCallable {
  CopyCallable(CopyCallable const& other)
      : value(other.value) {}

  CopyCallable(CopyCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  Ret operator()(Args&&...) { return Ret{value}; }

  explicit CopyCallable(bool x) : value(x)  {}
  Ret value;
};


template <class Ret = bool>
struct ConstCallable {
  ConstCallable(ConstCallable const& other)
      : value(other.value) {}

  ConstCallable(ConstCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  Ret operator()(Args&&...) const { return Ret{value}; }

  explicit ConstCallable(bool x) : value(x)  {}
  Ret value;
};



template <class Ret = bool>
struct NoExceptCallable {
  NoExceptCallable(NoExceptCallable const& other)
      : value(other.value) {}

  template <class ...Args>
  Ret operator()(Args&&...) noexcept { return Ret{value}; }

  template <class ...Args>
  Ret operator()(Args&&...) const noexcept { return Ret{value}; }

  explicit NoExceptCallable(bool x) : value(x)  {}
  Ret value;
};

struct CopyAssignableWrapper {
  CopyAssignableWrapper(CopyAssignableWrapper const&) = default;
  CopyAssignableWrapper(CopyAssignableWrapper&&) = default;
  CopyAssignableWrapper& operator=(CopyAssignableWrapper const&) = default;
  CopyAssignableWrapper& operator=(CopyAssignableWrapper &&) = default;

  template <class ...Args>
  bool operator()(Args&&...) { return value; }

  explicit CopyAssignableWrapper(bool x) : value(x) {}
  bool value;
};


struct MoveAssignableWrapper {
  MoveAssignableWrapper(MoveAssignableWrapper const&) = delete;
  MoveAssignableWrapper(MoveAssignableWrapper&&) = default;
  MoveAssignableWrapper& operator=(MoveAssignableWrapper const&) = delete;
  MoveAssignableWrapper& operator=(MoveAssignableWrapper &&) = default;

  template <class ...Args>
  bool operator()(Args&&...) { return value; }

  explicit MoveAssignableWrapper(bool x) : value(x) {}
  bool value;
};

struct MemFunCallable {
  explicit MemFunCallable(bool x) : value(x) {}

  bool return_value() const { return value; }
  bool return_value_nc() { return value; }
  bool value;
};

enum CallType : unsigned {
  CT_None,
  CT_NonConst = 1,
  CT_Const = 2,
  CT_LValue = 4,
  CT_RValue = 8
};

inline constexpr CallType operator|(CallType LHS, CallType RHS) {
    return static_cast<CallType>(static_cast<unsigned>(LHS) | static_cast<unsigned>(RHS));
}

struct ForwardingCallObject {

  template <class ...Args>
  bool operator()(Args&&...) & {
      set_call<Args&&...>(CT_NonConst | CT_LValue);
      return true;
  }

  template <class ...Args>
  bool operator()(Args&&...) const & {
      set_call<Args&&...>(CT_Const | CT_LValue);
      return true;
  }

  // Don't allow the call operator to be invoked as an rvalue.
  template <class ...Args>
  bool operator()(Args&&...) && {
      set_call<Args&&...>(CT_NonConst | CT_RValue);
      return true;
  }

  template <class ...Args>
  bool operator()(Args&&...) const && {
      set_call<Args&&...>(CT_Const | CT_RValue);
      return true;
  }

  template <class ...Args>
  static void set_call(CallType type) {
      assert(last_call_type == CT_None);
      assert(last_call_args == nullptr);
      last_call_type = type;
      last_call_args = &makeArgumentID<Args...>();
  }

  template <class ...Args>
  static bool check_call(CallType type) {
      bool result =
           last_call_type == type
        && last_call_args
        && *last_call_args == makeArgumentID<Args...>();
      last_call_type = CT_None;
      last_call_args = nullptr;
      return result;
  }

  static CallType      last_call_type;
  static TypeID const* last_call_args;
};

CallType ForwardingCallObject::last_call_type = CT_None;
TypeID const* ForwardingCallObject::last_call_args = nullptr;



///////////////////////////////////////////////////////////////////////////////
//                        BOOL TEST TYPES
///////////////////////////////////////////////////////////////////////////////

struct EvilBool {
  static int bang_called;

  EvilBool(EvilBool const&) = default;
  EvilBool(EvilBool&&) = default;

  friend EvilBool operator!(EvilBool const& other) {
    ++bang_called;
    return EvilBool{!other.value};
  }

private:
  friend struct MoveOnlyCallable<EvilBool>;
  friend struct CopyCallable<EvilBool>;
  friend struct NoExceptCallable<EvilBool>;

  explicit EvilBool(bool x) : value(x) {}
  EvilBool& operator=(EvilBool const& other) = default;

public:
  bool value;
};

int EvilBool::bang_called = 0;

struct ExplicitBool {
  ExplicitBool(ExplicitBool const&) = default;
  ExplicitBool(ExplicitBool&&) = default;

  explicit operator bool() const { return value; }

private:
  friend struct MoveOnlyCallable<ExplicitBool>;
  friend struct CopyCallable<ExplicitBool>;

  explicit ExplicitBool(bool x) : value(x) {}
  ExplicitBool& operator=(bool x) {
      value = x;
      return *this;
  }

  bool value;
};


struct NoExceptEvilBool {
  NoExceptEvilBool(NoExceptEvilBool const&) = default;
  NoExceptEvilBool(NoExceptEvilBool&&) = default;
  NoExceptEvilBool& operator=(NoExceptEvilBool const& other) = default;

  explicit NoExceptEvilBool(bool x) : value(x) {}

  friend NoExceptEvilBool operator!(NoExceptEvilBool const& other) noexcept {
    return NoExceptEvilBool{!other.value};
  }

  bool value;
};



void constructor_tests()
{
    {
        using T = MoveOnlyCallable<bool>;
        T value(true);
        using RetT = decltype(cuda::std::not_fn(cuda::std::move(value)));
        static_assert(cuda::std::is_move_constructible<RetT>::value, "");
        static_assert(!cuda::std::is_copy_constructible<RetT>::value, "");
        static_assert(!cuda::std::is_move_assignable<RetT>::value, "");
        static_assert(!cuda::std::is_copy_assignable<RetT>::value, "");
        auto ret = cuda::std::not_fn(cuda::std::move(value));
        // test it was moved from
        assert(value.value == false);
        // test that ret() negates the original value 'true'
        assert(ret() == false);
        assert(ret(0, 0.0, "blah") == false);
        // Move ret and test that it was moved from and that ret2 got the
        // original value.
        auto ret2 = cuda::std::move(ret);
        assert(ret() == true);
        assert(ret2() == false);
        assert(ret2(42) == false);
    }
    {
        using T = CopyCallable<bool>;
        T value(false);
        using RetT = decltype(cuda::std::not_fn(value));
        static_assert(cuda::std::is_move_constructible<RetT>::value, "");
        static_assert(cuda::std::is_copy_constructible<RetT>::value, "");
        static_assert(!cuda::std::is_move_assignable<RetT>::value, "");
        static_assert(!cuda::std::is_copy_assignable<RetT>::value, "");
        auto ret = cuda::std::not_fn(value);
        // test that value is unchanged (copied not moved)
        assert(value.value == false);
        // test 'ret' has the original value
        assert(ret() == true);
        assert(ret(42, 100) == true);
        // move from 'ret' and check that 'ret2' has the original value.
        auto ret2 = cuda::std::move(ret);
        assert(ret() == false);
        assert(ret2() == true);
        assert(ret2("abc") == true);
    }
    {
        using T = CopyAssignableWrapper;
        T value(true);
        T value2(false);
        using RetT = decltype(cuda::std::not_fn(value));
        static_assert(cuda::std::is_move_constructible<RetT>::value, "");
        static_assert(cuda::std::is_copy_constructible<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(cuda::std::is_move_assignable<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(cuda::std::is_copy_assignable<RetT>::value, "");
        auto ret = cuda::std::not_fn(value);
        assert(ret() == false);
        auto ret2 = cuda::std::not_fn(value2);
        assert(ret2() == true);
#if defined(_LIBCUDACXX_VERSION)
        ret = ret2;
        assert(ret() == true);
        assert(ret2() == true);
#endif // _LIBCUDACXX_VERSION
    }
    {
        using T = MoveAssignableWrapper;
        T value(true);
        T value2(false);
        using RetT = decltype(cuda::std::not_fn(cuda::std::move(value)));
        static_assert(cuda::std::is_move_constructible<RetT>::value, "");
        static_assert(!cuda::std::is_copy_constructible<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(cuda::std::is_move_assignable<RetT>::value, "");
        static_assert(!cuda::std::is_copy_assignable<RetT>::value, "");
        auto ret = cuda::std::not_fn(cuda::std::move(value));
        assert(ret() == false);
        auto ret2 = cuda::std::not_fn(cuda::std::move(value2));
        assert(ret2() == true);
#if defined(_LIBCUDACXX_VERSION)
        ret = cuda::std::move(ret2);
        assert(ret() == true);
#endif // _LIBCUDACXX_VERSION
    }
}

void return_type_tests()
{
    using cuda::std::is_same;
    {
        using T = CopyCallable<bool>;
        auto ret = cuda::std::not_fn(T{false});
        static_assert(is_same<decltype(ret()), bool>::value, "");
        static_assert(is_same<decltype(ret("abc")), bool>::value, "");
        assert(ret() == true);
    }
    {
        using T = CopyCallable<ExplicitBool>;
        auto ret = cuda::std::not_fn(T{true});
        static_assert(is_same<decltype(ret()), bool>::value, "");
        static_assert(is_same<decltype(ret(cuda::std::string("abc"))), bool>::value, "");
        assert(ret() == false);
    }
    {
        using T = CopyCallable<EvilBool>;
        auto ret = cuda::std::not_fn(T{false});
        static_assert(is_same<decltype(ret()), EvilBool>::value, "");
        EvilBool::bang_called = 0;
        auto value_ret = ret();
        assert(EvilBool::bang_called == 1);
        assert(value_ret.value == true);
        ret();
        assert(EvilBool::bang_called == 2);
    }
}

// Other tests only test using objects with call operators. Test various
// other callable types here.
void other_callable_types_test()
{
    { // test with function pointer
        auto ret = cuda::std::not_fn(returns_true);
        assert(ret() == false);
    }
    { // test with lambda
        auto returns_value = [](bool value) { return value; };
        auto ret = cuda::std::not_fn(returns_value);
        assert(ret(true) == false);
        assert(ret(false) == true);
    }
    { // test with pointer to member function
        MemFunCallable mt(true);
        const MemFunCallable mf(false);
        auto ret = cuda::std::not_fn(&MemFunCallable::return_value);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
    { // test with pointer to member function
        MemFunCallable mt(true);
        MemFunCallable mf(false);
        auto ret = cuda::std::not_fn(&MemFunCallable::return_value_nc);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
    { // test with pointer to member data
        MemFunCallable mt(true);
        const MemFunCallable mf(false);
        auto ret = cuda::std::not_fn(&MemFunCallable::value);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
}

void throws_in_constructor_test()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    struct ThrowsOnCopy {
      ThrowsOnCopy(ThrowsOnCopy const&) {
        throw 42;
      }
      ThrowsOnCopy() = default;
      bool operator()() const {
        assert(false);
#if defined(TEST_COMPILER_C1XX)
        __assume(0);
#else
        __builtin_unreachable();
#endif
      }
    };
    {
        ThrowsOnCopy cp;
        try {
            (void)cuda::std::not_fn(cp);
            assert(false);
        } catch (int const& value) {
            assert(value == 42);
        }
    }
#endif
}

void call_operator_sfinae_test() {
    { // wrong number of arguments
        using T = decltype(cuda::std::not_fn(returns_true));
        static_assert(cuda::std::is_invocable<T>::value, ""); // callable only with no args
        static_assert(!cuda::std::is_invocable<T, bool>::value, "");
    }
    { // violates const correctness (member function pointer)
        using T = decltype(cuda::std::not_fn(&MemFunCallable::return_value_nc));
        static_assert(cuda::std::is_invocable<T, MemFunCallable&>::value, "");
        static_assert(!cuda::std::is_invocable<T, const MemFunCallable&>::value, "");
    }
    { // violates const correctness (call object)
        using Obj = CopyCallable<bool>;
        using NCT = decltype(cuda::std::not_fn(Obj{true}));
        using CT = const NCT;
        static_assert(cuda::std::is_invocable<NCT>::value, "");
        static_assert(!cuda::std::is_invocable<CT>::value, "");
    }
    { // returns bad type with no operator!
        auto fn = [](auto x) { return x; };
        using T = decltype(cuda::std::not_fn(fn));
        static_assert(cuda::std::is_invocable<T, bool>::value, "");
        static_assert(!cuda::std::is_invocable<T, cuda::std::string>::value, "");
    }
}

void call_operator_forwarding_test()
{
    using Fn = ForwardingCallObject;
    auto obj = cuda::std::not_fn(Fn{});
    const auto& c_obj = obj;
    { // test zero args
        obj();
        assert(Fn::check_call<>(CT_NonConst | CT_LValue));
        cuda::std::move(obj)();
        assert(Fn::check_call<>(CT_NonConst | CT_RValue));
        c_obj();
        assert(Fn::check_call<>(CT_Const | CT_LValue));
        cuda::std::move(c_obj)();
        assert(Fn::check_call<>(CT_Const | CT_RValue));
    }
    { // test value categories
        int x = 42;
        const int cx = 42;
        obj(x);
        assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
        obj(cx);
        assert(Fn::check_call<const int&>(CT_NonConst | CT_LValue));
        obj(cuda::std::move(x));
        assert(Fn::check_call<int&&>(CT_NonConst | CT_LValue));
        obj(cuda::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_NonConst | CT_LValue));
        obj(42);
        assert(Fn::check_call<int&&>(CT_NonConst | CT_LValue));
    }
    { // test value categories - rvalue
        int x = 42;
        const int cx = 42;
        cuda::std::move(obj)(x);
        assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
        cuda::std::move(obj)(cx);
        assert(Fn::check_call<const int&>(CT_NonConst | CT_RValue));
        cuda::std::move(obj)(cuda::std::move(x));
        assert(Fn::check_call<int&&>(CT_NonConst | CT_RValue));
        cuda::std::move(obj)(cuda::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_NonConst | CT_RValue));
        cuda::std::move(obj)(42);
        assert(Fn::check_call<int&&>(CT_NonConst | CT_RValue));
    }
    { // test value categories - const call
        int x = 42;
        const int cx = 42;
        c_obj(x);
        assert(Fn::check_call<int&>(CT_Const | CT_LValue));
        c_obj(cx);
        assert(Fn::check_call<const int&>(CT_Const | CT_LValue));
        c_obj(cuda::std::move(x));
        assert(Fn::check_call<int&&>(CT_Const | CT_LValue));
        c_obj(cuda::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_Const | CT_LValue));
        c_obj(42);
        assert(Fn::check_call<int&&>(CT_Const | CT_LValue));
    }
    { // test value categories - const call rvalue
        int x = 42;
        const int cx = 42;
        cuda::std::move(c_obj)(x);
        assert(Fn::check_call<int&>(CT_Const | CT_RValue));
        cuda::std::move(c_obj)(cx);
        assert(Fn::check_call<const int&>(CT_Const | CT_RValue));
        cuda::std::move(c_obj)(cuda::std::move(x));
        assert(Fn::check_call<int&&>(CT_Const | CT_RValue));
        cuda::std::move(c_obj)(cuda::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_Const | CT_RValue));
        cuda::std::move(c_obj)(42);
        assert(Fn::check_call<int&&>(CT_Const | CT_RValue));
    }
    { // test multi arg
        const double y = 3.14;
        cuda::std::string s = "abc";
        obj(42, cuda::std::move(y), s, cuda::std::string{"foo"});
        Fn::check_call<int&&, const double&&, cuda::std::string&, cuda::std::string&&>(CT_NonConst | CT_LValue);
        cuda::std::move(obj)(42, cuda::std::move(y), s, cuda::std::string{"foo"});
        Fn::check_call<int&&, const double&&, cuda::std::string&, cuda::std::string&&>(CT_NonConst | CT_RValue);
        c_obj(42, cuda::std::move(y), s, cuda::std::string{"foo"});
        Fn::check_call<int&&, const double&&, cuda::std::string&, cuda::std::string&&>(CT_Const  | CT_LValue);
        cuda::std::move(c_obj)(42, cuda::std::move(y), s, cuda::std::string{"foo"});
        Fn::check_call<int&&, const double&&, cuda::std::string&, cuda::std::string&&>(CT_Const  | CT_RValue);
    }
}

void call_operator_noexcept_test()
{
    {
        using T = ConstCallable<bool>;
        T value(true);
        auto ret = cuda::std::not_fn(value);
        static_assert(!noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(!noexcept(cret()), "call should not be noexcept");
    }
    {
        using T = NoExceptCallable<bool>;
        T value(true);
        auto ret = cuda::std::not_fn(value);
        LIBCPP_STATIC_ASSERT(noexcept(!_CUDA_VSTD::__invoke(value)), "");
#if TEST_STD_VER > 14
        static_assert(noexcept(!cuda::std::invoke(value)), "");
#endif
        static_assert(noexcept(ret()), "call should be noexcept");
        auto const& cret = ret;
        static_assert(noexcept(cret()), "call should be noexcept");
    }
    {
        using T = NoExceptCallable<NoExceptEvilBool>;
        T value(true);
        auto ret = cuda::std::not_fn(value);
        static_assert(noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(noexcept(cret()), "call should not be noexcept");
    }
    {
        using T = NoExceptCallable<EvilBool>;
        T value(true);
        auto ret = cuda::std::not_fn(value);
        static_assert(!noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(!noexcept(cret()), "call should not be noexcept");
    }
}

void test_lwg2767() {
    // See https://cplusplus.github.io/LWG/lwg-defects.html#2767
    struct Abstract { virtual void f() const = 0; };
    struct Derived : public Abstract { void f() const {} };
    struct F { bool operator()(Abstract&&) { return false; } };
    {
        Derived d;
        Abstract &a = d;
        bool b = cuda::std::not_fn(F{})(cuda::std::move(a));
        assert(b);
    }
}

int main(int, char**)
{
    constructor_tests();
    return_type_tests();
    other_callable_types_test();
    throws_in_constructor_test();
    call_operator_sfinae_test(); // somewhat of an extension
    call_operator_forwarding_test();
    call_operator_noexcept_test();
    test_lwg2767();

  return 0;
}
