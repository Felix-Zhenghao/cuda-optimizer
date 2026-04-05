# 5.3. C++ Language Support

`nvcc` processes CUDA and device code according to the following specifications:

* **C++03** (ISO/IEC 14882:2003), `--std=c++03` flag.
* **C++11** (ISO/IEC 14882:2011), `--std=c++11` flag.
* **C++14** (ISO/IEC 14882:2014), `--std=c++14` flag.
* **C++17** (ISO/IEC 14882:2017), `--std=c++17` flag.
* **C++20** (ISO/IEC 14882:2020), `--std=c++20` flag.

Passing `nvcc` `-std=c++<version>` flag turns on all C++ features related to the specified version and also invokes the host preprocessor, compiler and linker with the corresponding C++ dialect option.

The compiler supports all language features of the supported standards, subject to the restrictions reported in the following sections.

## 5.3.1. C++11 Language Features

Table 34 C++11 Language Features Supported by NVCC for device code

| Language Feature | C++11 Proposal | NVCC/CUDA Toolkit 7.x |
| --- | --- | --- |
| [Rvalue references](#rvalue-references) | [N2118](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html) | ✓ |
|         Rvalue references for `*this` | [N2439](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm) | ✓ |
| Initialization of class objects by rvalues | [N1610](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html) | ✓ |
| Non-static data member initializers | [N2756](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm) | ✓ |
| Variadic templates | [N2242](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf) | ✓ |
|         Extending variadic template template parameters | [N2555](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf) | ✓ |
| [Initializer lists](#initializer-list) | [N2672](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm) | ✓ |
| Static assertions | [N1720](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html) | ✓ |
| `auto`-typed variables | [N1984](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf) | ✓ |
|         Multi-declarator `auto` | [N1737](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf) | ✓ |
|         Removal of auto as a storage-class specifier | [N2546](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm) | ✓ |
|         New function declarator syntax | [N2541](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm) | ✓ |
| [Lambda expressions](#lambda-expressions) | [N2927](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf) | ✓ |
| Declared type of an expression | [N2343](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf) | ✓ |
|         Incomplete return types | [N3276](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf) | ✓ |
| Right angle brackets | [N1757](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html) | ✓ |
| Default template arguments for function templates | [DR226](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226) | ✓ |
| Solving the SFINAE problem for expressions | [DR339](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html) | ✓ |
| Alias templates | [N2258](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf) | ✓ |
| Extern templates | [N1987](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm) | ✓ |
| Null pointer constant | [N2431](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf) | ✓ |
| Strongly-typed enums | [N2347](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf) | ✓ |
| Forward declarations for enums | [N2764](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf)   [DR1206](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206) | ✓ |
| Standardized attribute syntax | [N2761](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf) | ✓ |
| [Generalized constant expressions](#constexpr-functions) | [N2235](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf) | ✓ |
| Alignment support | [N2341](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf) | ✓ |
| Conditionally-supported behavior | [N1627](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf) | ✓ |
| Changing undefined behavior into diagnosable errors | [N1727](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf) | ✓ |
| Delegating constructors | [N1986](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf) | ✓ |
| Inheriting constructors | [N2540](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm) | ✓ |
| Explicit conversion operators | [N2437](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf) | ✓ |
| New character types | [N2249](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html) | ✓ |
| Unicode string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | ✓ |
| Raw string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | ✓ |
| Universal character names in literals | [N2170](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html) | ✓ |
| User-defined literals | [N2765](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf) | ✓ |
| Standard Layout Types | [N2342](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm) | ✓ |
| [Defaulted functions](#cpp11-defaulted-function) | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | ✓ |
| Deleted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | ✓ |
| Extended friend declarations | [N1791](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf) | ✓ |
| Extending `sizeof` | [N2253](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html)   [DR850](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850) | ✓ |
| [Inline namespaces](#inline-namespaces) | [N2535](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm) | ✓ |
| Unrestricted unions | [N2544](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf) | ✓ |
| [Local and unnamed types as template arguments](#templates) | [N2657](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm) | ✓ |
| Range-based for | [N2930](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html) | ✓ |
| Explicit `virtual` overrides | [N2928](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm)   [N3206](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm)   [N3272](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm) | ✓ |
| Minimal support for garbage collection and reachability-based leak detection | [N2670](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm) | ❌ |
| Allowing move constructors to throw [noexcept] | [N3050](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html) | ✓ |
| Defining move special member functions | [N3053](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html) | ✓ |
| **Concurrency** | | |
| Sequence points | [N2239](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html) | ❌ |
| Atomic operations | [N2427](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html) | ❌ |
| Strong Compare and Exchange | [N2748](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html) | ❌ |
| Bidirectional Fences | [N2752](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm) | ❌ |
| Memory model | [N2429](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm) | ❌ |
| Data-dependency ordering: atomics and memory model | [N2664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm) | ❌ |
| Propagating exceptions | [N2179](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html) | ❌ |
| Allow atomics use in signal handlers | [N2547](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm) | ❌ |
| Thread-local storage | [N2659](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm) | ❌ |
| Dynamic initialization and destruction with concurrency | [N2660](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm) | ❌ |
| **C99 Features in C++11** | | |
| `__func__` predefined identifier | [N2340](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm) | ✓ |
| C99 preprocessor | [N1653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm) | ✓ |
| `long long` | [N1811](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf) | ✓ |
| Extended integral types | [N1988](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf) | ❌ |

## 5.3.2. C++14 Language Features

Table 35 C++14 Language Features Supported by NVCC for device code

| Language Feature | C++14 Proposal | NVCC/CUDA Toolkit 9.x |
| --- | --- | --- |
| Tweak to certain C++ contextual conversions | [N3323](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf) | ✓ |
| Binary literals | [N3472](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf) | ✓ |
| [Functions with deduced return type](#return-type-deduction) | [N3638](https://isocpp.org/files/papers/N3638.html) | ✓ |
| Generalized lambda capture (init-capture) | [N3648](https://isocpp.org/files/papers/N3648.html) | ✓ |
| Generic (polymorphic) lambda expressions | [N3649](https://isocpp.org/files/papers/N3649.html) | ✓ |
| [Variable templates](#variable-templates) | [N3651](https://isocpp.org/files/papers/N3651.pdf) | ✓ |
| Relaxing requirements on constexpr functions | [N3652](https://isocpp.org/files/papers/N3652.html) | ✓ |
| Member initializers and aggregates | [N3653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html) | ✓ |
| Clarifying memory allocation | [N3664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html) | ❌ |
| Sized deallocation | [N3778](https://isocpp.org/files/papers/n3778.html) | ❌ |
| `[[deprecated]]` attribute | [N3760](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html) | ✓ |
| Single-quotation-mark as a digit separator | [N3781](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf) | ✓ |

## 5.3.3. C++17 Language Features

Table 36 C++17 Language Features Supported by NVCC for device code

| Language Feature | C++17 Proposal | NVCC/CUDA Toolkit 11.x |
| --- | --- | --- |
| Removing trigraphs | [N4086](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html) | ✓ |
| `u8` character literals | [N4267](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html) | ✓ |
| Folding expressions | [N4295](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html) | ✓ |
| Attributes for namespaces and enumerators | [N4266](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html) | ✓ |
| Nested namespace definitions | [N4230](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html) | ✓ |
| Allow constant evaluation for all non-type template arguments | [N4268](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html) | ✓ |
| Extending `static_assert` | [N3928](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf) | ✓ |
| New Rules for `auto` deduction from braced-init-list | [N3922](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html) | ✓ |
| Allow typename in a template template parameter | [N4051](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4051.html) | ✓ |
| `[[fallthrough]]` attribute | [P0188R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf) | ✓ |
| `[[nodiscard]]` attribute | [P0189R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf) | ✓ |
| `[[maybe_unused]]` attribute | [P0212R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf) | ✓ |
| Extension to aggregate initialization | [P0017R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html) | ✓ |
| Wording for `constexpr` lambda | [P0170R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf) | ✓ |
| Unary Folds and Empty Parameter Packs | [P0036R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf) | ✓ |
| Generalizing the Range-Based For Loop | [P0184R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html) | ✓ |
| Lambda capture of `*this` by Value | [P0018R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html) | ✓ |
| Construction Rules for `enum class` variables | [P0138R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf) | ✓ |
| Hexadecimal floating literals for C++ | [P0245R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html) | ✓ |
| Dynamic memory allocation for over-aligned data | [P0035R4](https://wg21.link/p0035) | ✓ |
| Guaranteed copy elision | [P0135R1](https://wg21.link/p0135) | ✓ |
| Refining Expression Evaluation Order for Idiomatic C++ | [P0145R3](https://wg21.link/p0145) | ✓ |
| `constexpr if` | [P0292R2](https://wg21.link/p0292) | ✓ |
| Selection statements with initializer | [P0305R1](https://wg21.link/p0305) | ✓ |
| Template argument deduction for class templates | [P0091R3](https://wg21.link/p0091)   [P0512R0](https://wg21.link/p0512r0) | ✓ |
| Declaring non-type template parameters with `auto` | [P0127R2](https://wg21.link/p0127) | ✓ |
| Using attribute namespaces without repetition | [P0028R4](https://wg21.link/p0028) | ✓ |
| Ignoring unsupported non-standard attributes | [P0283R2](https://wg21.link/p0283) | ✓ |
| [Structured bindings](#structured-binding) | [P0217R3](https://wg21.link/p0217) | ✓ |
| Remove Deprecated Use of the `register` Keyword | [P0001R1](https://wg21.link/p0001) | ✓ |
| Remove Deprecated `operator++(bool)` | [P0002R1](https://wg21.link/p0002) | ✓ |
| Make exception specifications be part of the type system | [P0012R1](https://wg21.link/p0012) | ✓ |
| `__has_include` for C++17 | [P0061R1](https://wg21.link/p0061) | ✓ |
| Rewording inheriting constructors (core issue 1941 et al) | [P0136R1](https://wg21.link/p0136) | ✓ |
| [Inline variables](#inline-variables) | [P0386R2](https://wg21.link/p0386r2) | ✓ |
| DR 150, Matching of template template arguments | [P0522R0](https://wg21.link/p0522r0) | ✓ |
| Removing dynamic exception specifications | [P0003R5](https://wg21.link/p0003r5) | ✓ |
| Pack expansions in using-declarations | [P0195R2](https://wg21.link/p0195r2) | ✓ |
| A `byte` type definition | [P0298R0](https://wg21.link/p0298r0) | ✓ |
| DR 727, In-class explicit instantiations | [CWG727](https://cplusplus.github.io/CWG/issues/727.html) | ✓ |

## 5.3.4. C++20 Language Features

GCC version ≥ 10.0, Clang version ≥ 10.0, Microsoft Visual Studio ≥ 2022, and nvc++ version ≥ 20.7.

Table 37 C++20 Language Features Supported by NVCC for device code

| Language Feature | C++20 Proposal | NVCC/CUDA Toolkit 12.x |
| --- | --- | --- |
| Default member initializers for bit-fields | [P0683R1](https://wg21.link/p0683r1) | ✓ |
| Fixing `const`-qualified pointers to members | [P0704R1](https://wg21.link/p0704r1) | ✓ |
| Allow lambda capture `[=, this]` | [P0409R2](https://wg21.link/p0409r2) | ✓ |
| `__VA_OPT__` for preprocessor comma elision | [P0306R4](https://wg21.link/p0306r4)   [P1042R1](https://wg21.link/p1042r1) | ✓ |
| Designated initializers | [P0329R4](https://wg21.link/p0329r4) | ✓ |
| Familiar template syntax for generic lambdas | [P0428R2](https://wg21.link/p0428r2) | ✓ |
| List deduction of vector | [P0702R1](https://wg21.link/p0702r1) | ✓ |
| Concepts | [P0734R0](https://wg21.link/p0734r0)   [P0857R0](https://wg21.link/p0857r0)   [P1084R2](https://wg21.link/p1084r2)   [P1141R2](https://wg21.link/p1141r2)   [P0848R3](https://wg21.link/p0848r3)   [P1616R1](https://wg21.link/p1616r1)   [P1452R2](https://wg21.link/p1452r2)   [P1972R0](https://wg21.link/p1972r0)   [P1980R0](https://wg21.link/p1980r0)   [P2092R0](https://wg21.link/p2092r0)   [P2103R0](https://wg21.link/p2103r0)   [P2113R0](https://wg21.link/p2113r0) | ✓ |
| Range-based for statements with initializer | [P0614R1](https://wg21.link/p0614r1) | ✓ |
| Simplifying implicit lambda capture | [P0588R1](https://wg21.link/p0588r1) | ✓ |
| ADL and function templates that are not visible | [P0846R0](https://wg21.link/p0846r0) | ✓ |
| `const` mismatch with defaulted copy constructor | [P0641R2](https://wg21.link/p0641r2) | ✓ |
| Less eager instantiation of `constexpr` functions | [P0859R0](https://wg21.link/p0859r0) | ✓ |
| [Consistent comparison](#cpp20-spaceship) (`operator<=>`) | [P0515R3](https://wg21.link/p0515r3)   [P0905R1](https://wg21.link/p0905r1)   [P1120R0](https://wg21.link/p1120r0)   [P1185R2](https://wg21.link/p1185r2)   [P1186R3](https://wg21.link/p1186r3)   [P1630R1](https://wg21.link/p1630r1)   [P1946R0](https://wg21.link/p1946r0)   [P1959R0](https://wg21.link/p1959r0)   [P2002R1](https://wg21.link/p2002r1)   [P2085R0](https://wg21.link/p2085r0) | ✓ |
| Access checking on specializations | [P0692R1](https://wg21.link/p0692r1) | ✓ |
| Default constructible and assignable stateless lambdas | [P0624R2](https://wg21.link/p0624r2) | ✓ |
| Lambdas in unevaluated contexts | [P0315R4](https://wg21.link/p0315r4) | ✓ |
| Language support for empty objects | [P0840R2](https://wg21.link/p0840r2) | ✓ |
| Relaxing the range-for loop customization point finding rules | [P0962R1](https://wg21.link/p0962r1) | ✓ |
| [Allow structured bindings to accessible members](#structured-binding) | [P0969R0](https://wg21.link/p0969r0) | ✓ |
| Relaxing the structured bindings customization point finding rules | [P0961R1](https://wg21.link/p0961r1) | ✓ |
| Down with typename! | [P0634R3](https://wg21.link/p0634r3) | ✓ |
| Allow pack expansion in lambda init-capture | [P0780R2](https://wg21.link/p0780r2)   [P2095R0](https://wg21.link/p2095r0) | ✓ |
| Proposed wording for `likely` and `unlikely` attributes | [P0479R5](https://wg21.link/p0479r5) | ✓ |
| Deprecate implicit capture of this via `[=]` | [P0806R2](https://wg21.link/p0806r2) | ✓ |
| Class Types in Non-Type Template Parameters | [P0732R2](https://wg21.link/p0732r2) | ✓ |
| Inconsistencies with non-type template parameters | [P1907R1](https://wg21.link/p1907r1) | ✓ |
| Atomic Compare-and-Exchange with Padding Bits | [P0528R3](https://wg21.link/p0528r3) | ✓ |
| Efficient sized delete for variable sized classes | [P0722R3](https://wg21.link/p0722r3) | ✓ |
| Allowing Virtual Function Calls in Constant Expressions | [P1064R0](https://wg21.link/p1064r0) | ✓ |
| Prohibit aggregates with user-declared constructors | [P1008R1](https://wg21.link/p1008r1) | ✓ |
| `explicit(bool)` | [P0892R2](https://wg21.link/p0892r2) | ✓ |
| Signed integers are two's complement | [P1236R1](https://wg21.link/p1236r1) | ✓ |
| `char8_t` | [P0482R6](https://wg21.link/p0482r6) | ✓ |
| [Immediate functions](#cpp20-consteval) (`consteval`) | [P1073R3](https://wg21.link/p1073r3)   [P1937R2](https://wg21.link/p1937r2) | ✓ |
| `std::is_constant_evaluated` | [P0595R2](https://wg21.link/p0595r2) | ✓ |
| Nested `inline` namespaces | [P1094R2](https://wg21.link/p1094r2) | ✓ |
| Relaxations of `constexpr` restrictions | [P1002R1](https://wg21.link/p1002r1)   [P1327R1](https://wg21.link/p1327r1)   [P1330R0](https://wg21.link/p1330r0)   [P1331R2](https://wg21.link/p1331r2)   [P1668R1](https://wg21.link/p1668r1)   [P0784R7](https://wg21.link/p0784r7) | ✓ |
| Feature test macros | [P0941R2](https://wg21.link/p0941r2) | ✓ |
| Modules | [P1103R3](https://wg21.link/p1103r3)   [P1766R1](https://wg21.link/p1766r1)   [P1811R0](https://wg21.link/p1811r0)   [P1703R1](https://wg21.link/p1703r1)   [P1874R1](https://wg21.link/p1874r1)   [P1979R0](https://wg21.link/p1979r0)   [P1779R3](https://wg21.link/p1779r3)   [P1857R3](https://wg21.link/p1857r3)   [P2115R0](https://wg21.link/p2115r0)   [P1815R2](https://wg21.link/p1815r2) | ❌ |
| Coroutines | [P0912R5](https://wg21.link/p0912r5) | ❌ |
| Parenthesized initialization of aggregates | [P0960R3](https://wg21.link/p0960r3)   [P1975R0](https://wg21.link/p1975r0) | ✓ |
| DR: array size deduction in new-expression | [P1009R2](https://wg21.link/p1009r2) | ✓ |
| DR: Converting from `T*` to bool should be considered narrowing | [P1957R2](https://wg21.link/p1957r2) | ✓ |
| Stronger Unicode requirements | [P1041R4](https://wg21.link/p1041r4)   [P1139R2](https://wg21.link/p1139r2) | ✓ |
| Structured binding extensions | [P1091R3](https://wg21.link/p1091r3)   [P1381R1](https://wg21.link/p1381r1) | ✓ |
| Deprecate `a[b,c]` | [P1161R3](https://wg21.link/p1161r3) | ✓ |
| Deprecating some uses of `volatile` | [P1152R4](https://wg21.link/p1152r4) | ✓ |
| `[[nodiscard("with reason")]]` | [P1301R4](https://wg21.link/p1301r4) | ✓ |
| `using enum` | [P1099R5](https://wg21.link/p1099r5) | ✓ |
| Class template argument deduction for aggregates | [P1816R0](https://wg21.link/p1816r0)   [P2082R1](https://wg21.link/p2082r1) | ✓ |
| Class template argument deduction for alias templates | [P1814R0](https://wg21.link/p1814r0) | ✓ |
| Permit conversions to arrays of unknown bound | [P0388R4](https://wg21.link/p0388r4) | ✓ |
| `constinit` | [P1143R2](https://wg21.link/p1143r2) | ✓ |
| Layout-compatibility and Pointer-interconvertibility Traits | [P0466R5](https://wg21.link/p0466r5) | ✓ |
| DR: Checking for abstract class types | [P0929R2](https://wg21.link/p0929r2) | ✓ |
| DR: More implicit moves | [P1825R0](https://wg21.link/p1825r0) | ✓ |
| DR: Pseudo-destructors end object lifetimes | [P0593R6](https://wg21.link/p0593r6) | ✓ |

## 5.3.5. CUDA C++ Standard Library

CUDA provides an implementation of the C++ Standard Library (STL), called [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html). The library presents the following benefits:

* The functionalities are available on both host and device.
* Compatible with all [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2) platforms supported by the CUDA Toolkit.
* Compatible with all [GPU architectures](https://developer.nvidia.com/cuda-gpus) supported by the last two major versions of the CUDA Toolkit.
* Compatible with all [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with the current and previous major versions.
* Provides C++17 backports of C++ Standard Library features available in recent standard versions, including C++20, C++23, and C++26.
* Supports extended data types, such as 128-bit integers (`__int128`), half-precision floats (`__half`), Bfloat16 (`__nv_bfloat16`), and quad-precision floats (`__float128`).
* Highly optimized for device code.

In addition, `libcu++` provides [extended features](https://nvidia.github.io/cccl/libcudacxx/extended_api.html) that are not available in the C++ Standard Library to improve productivity and application performance. Such features include mathematical functions, memory operations, synchronization primitives, container extensions, high-level abstractions of CUDA intrinsics, C++ PTX wrappers, and more.

`libcu++` is available as part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), as well as part of the open-source [CCCL](https://nvidia.github.io/cccl/) repository.

## 5.3.6. C Standard Library Functions

### 5.3.6.1. `clock()` and `clock64()`

```
__host__ __device__ clock_t   clock();
__device__          long long clock64();
```

When executed in device code, it returns the value of a per-multiprocessor counter that increments every clock cycle. Sampling this counter at the beginning and end of a kernel, subtracting the two values, and recording the result for each thread provides an estimate of the number of clock cycles the device spends executing the thread. However, this value does not represent the actual number of clock cycles the device spends executing the thread's instructions. The former number is greater than the latter because threads are time-sliced.

Hint

* The corresponding [CUDA C++ function](https://en.cppreference.com/w/cpp/chrono/c/clock.html) `cuda::std::clock()` is provided in the `<cuda/std/ctime>` header.
* A portable [C++](https://en.cppreference.com/w/cpp/header/chrono) `<chrono>` implementation is also provided in the `<cuda/std/chrono>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time) for similar purposes.

### 5.3.6.2. `printf()`

```
int printf(const char* format[, arg, ...]);
```

The function prints formatted output from a kernel to a host-side output stream.

The in-kernel `printf()` function behaves similarly to the standard C library `printf()` function. Users should refer to their host system's manual pages for complete descriptions of `printf()` behavior. Essentially, the string passed in as `format` is output to a stream on the host.

The `printf()` command is executed like any other device-side function: per thread and in the context of the calling thread. In a multi-threaded kernel, a straightforward call to `printf()` will be executed by every thread using the data specified by that thread. Consequently, multiple versions of the output string will appear at the host stream, each corresponding to a thread that encountered the `printf()`.

Unlike the C standard `printf()`, which returns the number of characters printed, CUDA's `printf()` returns the number of arguments parsed. If no arguments follow the format string, 0 is returned. If the format string is `NULL`, `-1` is returned. If an internal error occurs, -2 is returned.

Internally, `printf()` uses a shared data structure, so it is possible that calling `printf()` may alter the execution order of threads. In particular, a thread that calls `printf()` might take a longer execution path than a thread that does not call `printf()`, and the length of that path depends on the parameters of `printf()`. However, note that CUDA makes no guarantees about the order of thread execution except at explicit `__syncthreads()` barriers. Therefore, it is impossible to tell whether the order of execution has been modified by `printf()` or by other scheduling behaviors in the hardware.

---

**Format Specifiers**

As for standard `printf()`, format specifiers take the form: `%[flags][width][.precision][size]type`

The following fields are supported. See the widely available documentation for a complete description of all behaviors.

* Flags: `#`, `' '`, `0`, `+`, `-`
* Width: `*`, `0-9`
* Precision: `0-9`
* Size: `h`, `l`, `ll`
* Type: `%cdiouxXpeEfgGaAs`

---

**Limitations**

The final formatting of the `printf()` output takes place on the host system. This means that the format string must be understood by the compiler and C library of the host system. While every effort has been made to ensure that the format specifiers supported by CUDA's `printf()` function are a universal subset of those supported by the most common host compilers, the exact behavior will be dependent on the host operating system.

`printf()` accepts all valid combinations of flags and types. This is because it cannot determine what will and will not be valid on the host system where the final output is formatted. Consequently, output may be undefined if the program emits a format string containing invalid combinations.

The `printf()` function can accept up to 32 arguments, in addition to the format string. Any additional arguments will be ignored, and the format specifier will be output as is.

Due to the different sizes of the `long` type on Windows platforms (32-bit) and Linux platforms (64-bit), a kernel compiled on a Linux machine and then run on a Windows machine will produce corrupted output for all format strings that include `%ld`. To ensure safety, it is recommended that the compilation and execution platforms match.

---

**Host-Side Buffer**

The output buffer for `printf()` is set to a fixed size before kernel launch. The buffer is circular, so if more output is produced during kernel execution than can fit in the buffer, older output is overwritten. The buffer is flushed only when one of the following actions is performed:

* Kernel launch via `<<< >>>` or `cuLaunchKernel()`: at the start of the launch, and if the `CUDA_LAUNCH_BLOCKING` environment variable is set to 1, at the end of the launch as well,
* Synchronization via `cudaDeviceSynchronize()`, `cuCtxSynchronize()`, `cudaStreamSynchronize()`, `cuStreamSynchronize()`, `cudaEventSynchronize()`, or `cuEventSynchronize()`,
* Memory copies via any blocking version of `cudaMemcpy*()` or `cuMemcpy*()`,
* Module loading/unloading via `cuModuleLoad()` or `cuModuleUnload()`,
* Context destruction via `cudaDeviceReset()` or `cuCtxDestroy()`.
* Prior to executing a stream callback added by `cudaLaunchHostFunc()` or `cuLaunchHostFunc()`.

Note that the buffer is not automatically flushed when the program exits.

The following API functions set and retrieve the size of the buffer used to transfer `printf()` arguments and internal metadata to the host. The default size is one megabyte.

* `cudaDeviceGetLimit(size_t* size,cudaLimitPrintfFifoSize)`
* `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)`

---

**Examples**

The following code sample:

```
#include <stdio.h>

__global__ void helloCUDA(float value) {
    printf("Hello thread %d, value=%f\n", threadIdx.x, value);
}

int main() {
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

will output:

```
Hello thread 2, value=1.2345
Hello thread 1, value=1.2345
Hello thread 4, value=1.2345
Hello thread 0, value=1.2345
Hello thread 3, value=1.2345
```

Notice that each thread encounters the `printf()` command. Therefore, there are as many lines of output as there are threads in the grid.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/d4MPj7qG8).

---

The following code sample:

```
#include <stdio.h>

__global__ void helloCUDA(float value) {
    if (threadIdx.x == 0)
        printf("Hello thread %d, value=%f\n", threadIdx.x, value);
}

int main() {
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

will output:

```
Hello thread 0, value=1.2345
```

Clearly, the `if()` statement limits which threads call `printf()`, so only one line of output is seen.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/YqEss81sf).

### 5.3.6.3. `memcpy()` and `memset()`

```
__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
```

The function copies `size` bytes from the memory location pointed by `src` to the memory location pointed by `dest`.

```
__host__ __device__ void* memset(void* ptr, int value, size_t size);
```

The function sets `size` bytes of memory block pointed by `ptr` to `value`, interpreted as an `unsigned char`.

Hint

It is suggested to use the `cuda::std::memcpy()` and `cuda::std::memset()` functions provided in the `<cuda/std/cstring>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring) as safer versions of `memcpy` and `memset`.

### 5.3.6.4. `malloc()` and `free()`

```
__host__ __device__ void* malloc(size_t size);
// or cuda::std::malloc(), cuda::std::calloc() in the <cuda/std/cstdlib> header
```

The functions `malloc()` (device-side), `cuda::std::malloc()`, and `cuda::std::calloc()` allocate at least `size` bytes from the device heap and return a pointer to the allocated memory. If insufficient memory exists to fulfill the request, it returns `NULL`. The returned pointer is guaranteed to be aligned to a 16-byte boundary.

```
__device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
// or cuda::std::aligned_alloc() in the <cuda/std/cstdlib> header
```

The functions `__nv_aligned_device_malloc()` and [C++](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc) `cuda::std::aligned_alloc()` allocate at least `size` bytes from the device heap and return a pointer to the allocated memory. If there is insufficient memory to fulfill the requested size or alignment, it returns `NULL`. The address of the allocated memory is a multiple of `align`. `align` must be a non-zero power of two.

```
__host__ __device__ void free(void* ptr);
// or cuda::std::free() in the <cuda/std/cstdlib> header
```

The device-side functions `free()` and `cuda::std::free()` deallocate the memory pointed to by `ptr`, which must have been returned by a previous call to `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()`. If `ptr` is `NULL`, the call to `free()` or `cuda::std::free()` is ignored. Repeated calls to `free()` or `cuda::std::free()` with the same `ptr` have undefined behavior.

Memory allocated by a given CUDA thread via `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`,
`__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()` remain allocated for the lifetime of the CUDA context, or until it is explicitly released by a call to `free()` or `cuda::std::free()`. This memory can be used by other CUDA threads, even those from subsequent kernel launches. Any CUDA thread can free memory allocated by another thread; however, care should be taken to ensure that the same pointer is not freed more than once.

---

**Heap Memory API**

The size of the device memory heap must be specified before any program that allocates or frees memory in device code, including the `new` and `delete` keywords. If any program uses the device memory heap without explicitly specifying the heap size, a default heap of eight megabytes is allocated.

The following API functions get and set the heap size:

* `cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)`
* `cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)`

The heap size granted will be at least `size` bytes. [cuCtxGetLimit()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8) and [cudaDeviceGetLimit()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2) return the currently requested heap size.

The actual memory allocation for the heap occurs when a module is loaded into the context, either explicitly through the CUDA driver API (see [Module](../03-advanced/driver-api.html#driver-api-module)) or implicitly through the CUDA runtime API. If memory allocation fails, the module load generates a `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` error.

The heap size cannot be changed after a module has been loaded, and it does not dynamically resize according to need.

The memory reserved for the device heap is in addition to the memory allocated through host-side CUDA API calls such as `cudaMalloc()`.

---

**Interoperability with the Host Memory API**

Memory allocated via the device-side functions `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, `cuda::std::aligned_alloc()`, or the `new` keyword cannot be used or freed with runtime or driver API calls such as `cudaMalloc`, `cudaMemcpy`, or `cudaMemset`. Similarly, memory allocated via the host runtime API cannot be freed using the device-side functions `free()`, `cuda::std::free()`, or the `delete` keyword.

---

Per-Thread Allocation example:

```
#include <stdlib.h>
#include <stdio.h>

__global__ void single_thread_allocation_kernel() {
    size_t size = 123;
    char*  ptr  = (char*) malloc(size);
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main() {
    // Set a heap size of 128 megabytes.
    // Note that this must be done before any kernel is launched.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    single_thread_allocation_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

will output:

```
Thread 0 got pointer: 0x20d5ffe20
Thread 1 got pointer: 0x20d5ffec0
Thread 2 got pointer: 0x20d5fff60
Thread 3 got pointer: 0x20d5f97c0
Thread 4 got pointer: 0x20d5f9720
```

Notice how each thread encounters the `malloc()` and `memset()` commands and so receives and initializes its own allocation.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/z7K191z58).

---

Per-Thread-Block Allocation example:

```
#include <stdlib.h>

__global__ void block_level_allocation_kernel() {
    __shared__ int* data;
    // The first thread in the block performs the allocation and shares the pointer
    // with all other threads through shared memory, so that access can be coalesced.
    if (threadIdx.x == 0) {
        size_t size = blockDim.x * 64; // 64 bytes per thread are allocated.
        data = (int*) malloc(size);
    }
    __syncthreads();
    // Check for failure
    if (data == nullptr)
        return;

    // Threads index into the memory, ensuring coalescence
    for (int i = 0; i < 64; ++i)
        data[i * blockDim.x + threadIdx.x] = threadIdx.x;
    // Ensure all threads complete before freeing
    __syncthreads();

    // Only one thread may free the memory!
    if (threadIdx.x == 0)
        free(data);
}

int main() {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    block_level_allocation_kernel<<<10, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/7s8x7oonz).

---

Allocation Persisting Between Kernel Launches example:

```
#include <stdlib.h>
#include <stdio.h>

const int NUM_BLOCKS = 20;

__device__ int* data_ptrs[NUM_BLOCKS]; // Per-block pointer

__global__ void allocate_memory_kernel() {
    // Only the first thread in the block performs the allocation
    // since we need only one allocation per block.
    if (threadIdx.x == 0)
        data_ptrs[blockIdx.x] = (int*) malloc(blockDim.x * 4);
    __syncthreads();
    // Check for failure
    if (data_ptrs[blockIdx.x] == nullptr)
        return;
    // Zero the data with all threads in parallel
    data_ptrs[blockIdx.x][threadIdx.x] = 0;
}

// Simple example: store the thread ID into each element
__global__ void use_memory_kernel() {
    int* ptr = data_ptrs[blockIdx.x];
    if (ptr != nullptr)
        ptr[threadIdx.x] += threadIdx.x;
}

// Print the content of the buffer before freeing it
__global__ void free_memory_kernel() {
    int* ptr = data_ptrs[blockIdx.x];
    if (ptr != nullptr)
        printf("Block %d, Thread %d: final value = %d\n",
            blockIdx.x, threadIdx.x, ptr[threadIdx.x]);
    // Only free from one thread!
    if (threadIdx.x == 0)
        free(ptr);
}

int main() {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    // Allocate memory
    allocate_memory_kernel<<<NUM_BLOCKS, 10>>>();

    // Use memory
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();

    // Free memory
    free_memory_kernel<<<NUM_BLOCKS, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/h7r6G3dGP).

### 5.3.6.5. `alloca()`

```
__host__ __device__ void* alloca(size_t size);
```

The `alloca()` function allocates `size` bytes of memory within the caller's stack frame. The returned value is a pointer to the allocated memory. When the function is invoked from device code, the beginning of the memory is 16-byte aligned. The memory is automatically freed when the caller returns from `alloca()`.

Note

On the Windows platform, the `<malloc.h>` header file must be included before using the `alloca()` function. Calls to `alloca()` may cause the stack to overflow; the user needs to adjust the stack size accordingly.

Example:

```
__device__ void device_function(int num_items) {
    int4* ptr = (int4*) alloca(num_items * sizeof(int4));
    // use of ptr
    ...
}
```

