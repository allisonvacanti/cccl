//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_INTEGER_SEQUENCE_H
#define _LIBCUDACXX___UTILITY_INTEGER_SEQUENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t...>
struct __tuple_indices
{};

template <class _IdxType, _IdxType... _Values>
struct __integer_sequence
{
  template <template <class _OIdxType, _OIdxType...> class _ToIndexSeq, class _ToIndexType>
  using __convert = _ToIndexSeq<_ToIndexType, _Values...>;

  template <size_t _Sp>
  using __to_tuple_indices = __tuple_indices<(_Values + _Sp)...>;
};

#if !__has_builtin(__make_integer_seq) || defined(_LIBCUDACXX_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE)

namespace __detail
{

template <typename _Tp, size_t... _Extra>
struct __repeat;
template <typename _Tp, _Tp... _Np, size_t... _Extra>
struct __repeat<__integer_sequence<_Tp, _Np...>, _Extra...>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE __integer_sequence<
    _Tp,
    _Np...,
    sizeof...(_Np) + _Np...,
    2 * sizeof...(_Np) + _Np...,
    3 * sizeof...(_Np) + _Np...,
    4 * sizeof...(_Np) + _Np...,
    5 * sizeof...(_Np) + _Np...,
    6 * sizeof...(_Np) + _Np...,
    7 * sizeof...(_Np) + _Np...,
    _Extra...>
    type;
};

template <size_t _Np>
struct __parity;
template <size_t _Np>
struct __make : __parity<_Np % 8>::template __pmake<_Np>
{};

template <>
struct __make<0>
{
  typedef __integer_sequence<size_t> type;
};
template <>
struct __make<1>
{
  typedef __integer_sequence<size_t, 0> type;
};
template <>
struct __make<2>
{
  typedef __integer_sequence<size_t, 0, 1> type;
};
template <>
struct __make<3>
{
  typedef __integer_sequence<size_t, 0, 1, 2> type;
};
template <>
struct __make<4>
{
  typedef __integer_sequence<size_t, 0, 1, 2, 3> type;
};
template <>
struct __make<5>
{
  typedef __integer_sequence<size_t, 0, 1, 2, 3, 4> type;
};
template <>
struct __make<6>
{
  typedef __integer_sequence<size_t, 0, 1, 2, 3, 4, 5> type;
};
template <>
struct __make<7>
{
  typedef __integer_sequence<size_t, 0, 1, 2, 3, 4, 5, 6> type;
};

template <>
struct __parity<0>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type>
  {};
};
template <>
struct __parity<1>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 1>
  {};
};
template <>
struct __parity<2>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 2, _Np - 1>
  {};
};
template <>
struct __parity<3>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 3, _Np - 2, _Np - 1>
  {};
};
template <>
struct __parity<4>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 4, _Np - 3, _Np - 2, _Np - 1>
  {};
};
template <>
struct __parity<5>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1>
  {};
};
template <>
struct __parity<6>
{
  template <size_t _Np>
  struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1>
  {};
};
template <>
struct __parity<7>
{
  template <size_t _Np>
  struct __pmake
      : __repeat<typename __make<_Np / 8>::type, _Np - 7, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1>
  {};
};

} // namespace __detail

#endif // !__has_builtin(__make_integer_seq) || defined(_LIBCUDACXX_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE)

#if __has_builtin(__make_integer_seq)
template <size_t _Ep, size_t _Sp>
using __make_indices_imp =
  typename __make_integer_seq<__integer_sequence, size_t, _Ep - _Sp>::template __to_tuple_indices<_Sp>;
#else
template <size_t _Ep, size_t _Sp>
using __make_indices_imp = typename __detail::__make<_Ep - _Sp>::type::template __to_tuple_indices<_Sp>;

#endif

#if _CCCL_STD_VER > 2011

template <class _Tp, _Tp... _Ip>
struct _LIBCUDACXX_TEMPLATE_VIS integer_sequence
{
  typedef _Tp value_type;
  static_assert(is_integral<_Tp>::value, "std::integer_sequence can only be instantiated with an integral type");
  static _LIBCUDACXX_INLINE_VISIBILITY constexpr size_t size() noexcept
  {
    return sizeof...(_Ip);
  }
};

template <size_t... _Ip>
using index_sequence = integer_sequence<size_t, _Ip...>;

#  if __has_builtin(__make_integer_seq) && !defined(_LIBCUDACXX_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE)

template <class _Tp, _Tp _Ep>
using __make_integer_sequence _LIBCUDACXX_NODEBUG_TYPE = __make_integer_seq<integer_sequence, _Tp, _Ep>;

#  else

template <typename _Tp, _Tp _Np>
using __make_integer_sequence_unchecked _LIBCUDACXX_NODEBUG_TYPE =
  typename __detail::__make<_Np>::type::template __convert<integer_sequence, _Tp>;

template <class _Tp, _Tp _Ep>
struct __make_integer_sequence_checked
{
  static_assert(is_integral<_Tp>::value, "std::make_integer_sequence can only be instantiated with an integral type");
  static_assert(0 <= _Ep, "std::make_integer_sequence must have a non-negative sequence length");
  // Workaround GCC bug by preventing bad installations when 0 <= _Ep
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68929
  typedef _LIBCUDACXX_NODEBUG_TYPE __make_integer_sequence_unchecked<_Tp, 0 <= _Ep ? _Ep : 0> type;
};

template <class _Tp, _Tp _Ep>
using __make_integer_sequence _LIBCUDACXX_NODEBUG_TYPE = typename __make_integer_sequence_checked<_Tp, _Ep>::type;

#  endif

template <class _Tp, _Tp _Np>
using make_integer_sequence = __make_integer_sequence<_Tp, _Np>;

template <size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;

template <class... _Tp>
using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_INTEGER_SEQUENCE_H
