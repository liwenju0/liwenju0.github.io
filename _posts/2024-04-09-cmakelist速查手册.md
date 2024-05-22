---
layout: post
title:  "cmakelist速查手册"
date:   2024-04-09 11:20:08 +0800
category: "计算机基础"
published: true
---

原始文档链接：[cmakelist-cheatsheet](https://usercontent.one/wp/cheatsheet.czutro.ch/wp-content/uploads/2020/09/CMake_Cheatsheet.pdf)

<!--more-->

## CMake 速查表 - 简明介绍 CMake

这份速查表将给你一个关于 CMake 如何工作以及如何使用它来配置软件项目的想法。文档和 CMake 示例可以在 https://github.com/mortennobel/CMake-Cheatsheet 找到。

## CMake - 创建一个简单的 C++ 项目

CMake 是一个工具，用于配置跨平台源代码项目应该如何在给定平台上构建。一个小项目可能像这样组织：
例1:
```
CMakeLists.txt 
src/main.cpp 
src/foo.cpp 
src/foo.hpp
```


这个项目包含两个位于 src 目录的源文件和同一个目录中的一个头文件。在运行 CMake 时，你会被要求提供一个二进制目录。最好的做法是创建一个新目录，因为这个目录将包含所有与构建项目相关的文件。如果出现问题，你可以删除文件夹并重新开始。运行 CMake 不会创建最终的可执行文件，而是会生成 Visual Studio、XCode 或 make 文件的项目文件。使用这些工具来构建项目。

## 理解 CMakeLists.txt

使用 CMake 创建项目文件需要一个 CMakeLists.txt 文件，该文件描述了项目的架构以及应该如何构建。例1 的文件看起来像这样：

```cmake
cmake_minimum_required (VERSION 2.9)
# 设置项目名称
project ( HelloProject )
# 编译并链接 main.cpp 和 foo.cpp 成可执行文件 Hello
add_executable ( Hello src/main.cpp src/foo.cpp )
```
首先，定义了 CMake 的最小版本。

然后使用 project() 命令定义项目名称。

一个项目可以包含多个目标（可执行文件或库）。这个项目定义了一个名为 Hello 的单个可执行目标，它通过编译和链接两个源文件 main.cpp 和 foo.cpp 来创建。

当这两个源文件被编译时，编译器会搜索头文件 foo.h，因为两个源文件都使用了
```c
#include "foo.hpp" 
```
由于文件位于与源文件相同的位置，编译器不会遇到任何问题。

# CMake 脚本语言
CMakeLists.txt 使用基于命令的编程语言描述构建过程。命令不区分大小写，并接受一系列参数。

```cmake
# 这是一个注释。
COMMAND( arguments go here )
ANOTHER_COMMAND() # 这个命令目前还没有参数 
YET_ANOTHER_COMMAND( these
arguments are spread # 另一个注释
over several lines )
```
CMake 脚本还有变量。变量可以由 CMake 定义，也可以在 CMake 脚本中定义。

```set(parameter value) ```
命令将给定参数设置为一个值。

```message(value)``` 命令将值打印到控制台。

要获取变量的值，请使用 ```${varname}```，它会将变量名替换为其值。
```cmake
cmake_minimum_required (VERSION 2.9)
SET( x 3 ) # x = "3"
SET( y 1 ) # y = "1"
MESSAGE( x y ) # 显示 "xy"
MESSAGE( ${x}${y} ) # 显示 "31"
```
所有变量值都是文本字符串。

文本字符串可以作为布尔表达式评估（例如，在 IF() 和 WHILE() 中使用）。

值 "FALSE"、"OFF"、"NO" 或任何以 "-NOTFOUND" 结尾的字符串将被评估为 false，其他所有内容都为 true。

文本字符串可以通过使用分号分隔实体来表示多个值的列表。
```cmake
cmake_minimum_required (VERSION 2.9)
SET( x 3 2) # x = "3;2"
SET( y hello world! ) # y = "hello; world!"
SET( z "hello world!" ) # y = "hello world!"
MESSAGE( ${x} ) # 打印 "3;2"
MESSAGE( "y = ${y} z = ${z}") # 打印 y = hello; world! z = hello world!


列表可以使用 FOREACH (var val) 命令进行迭代。

```cmake
cmake_minimum_required (VERSION 2.9)
SET( x 3 2) # x = "3;2"
FOREACH ( val ${x}) 
  MESSAGE( ${val } )
ENDFOREACH( val )
# 打印 
# 3 
# 2
```
## 暴露编译选项
CMake 允许最终用户（运行 CMake 的人）修改一些变量的值。

这通常用于定义构建的属性，如文件位置、机器架构和字符串值。
```set(<variable> <value> CACHE <type> <docstring>)```
命令将变量设置为默认值，但允许 CMake 用户在配置构建时更改该值。类型应该是以下之一：

- FILEPATH = 文件选择器对话框。
- PATH = 目录选择器对话框。
- STRING = 任意字符串。
- BOOL = 布尔 ON/OFF 复选框。
- INTERNAL = 无 GUI 条目（用于持久变量）。
  
在以下示例中，用户可以配置是否应该打印 "Hello" 或基于配置变量 hello 和 other_msg 的替代字符串。
```cmake
cmake_minimum_required (VERSION 2.9)
SET( hello true CACHE BOOL "If true write hello ")
SET( other_msg "Hi" CACHE STRING "Not hello value ")
IF ( ${hello} )
  MESSAGE( "Hello" )
ELSE ( ${hello} )
  MESSAGE( ${other_msg} )
ENDIF ( ${hello} )
```
在配置项目期间，CMake 用户将被提示选择暴露的选项。
![2024-04-09-14-43-51](https://raw.githubusercontent.com/liwenju0/blog_pictures/master/pics/2024-04-09-14-43-51.png)

CMake 用户输入的值将被保存在文本文件 CMakeCache.txt 中，作为键值对：
```cmake
// ....
// 打印 hello
hello:BOOL=OFF
// Not hello value
other_msg:STRING=Guten tag // ....
```
## 复杂项目
一些项目既包含多个可执行文件，也包含多个库。例如，当同时拥有单元测试和程序时。通常将这些子项目分离到子文件夹中。示例：

```
CMakeLists.txt 

somelib/CMakeLists.txt 
somelib/foo.hpp 
somelib/foo.cpp 

someexe/CMakeLists.txt 
someexe/main.cpp
```
主 CMakeLists.txt 包含基本的项目设置，然后包括子项目：

```cmake
# CMakeLists.txt
cmake_minimum_required (VERSION 2.9)
# 设置项目名称
project ( HelloProject )
add_subdirectory ( somelib )
add_subdirectory ( someexe )
```
首先，库 Foo 从 somelib 目录的源代码编译：

```cmake
# somelib/CMakeLists.txt
# 编译并链接 foo.cpp
add_library (Foo STATIC foo.cpp )
```
最后，可执行文件 Hello 被编译并链接到 Foo 库 - 请注意这里使用的是目标名称，而不是实际路径。
由于 main.cpp 引用了头文件 Foo.hpp，somelib 目录被添加到头文件搜索路径：

```cmake
# someexe/CMakeLists.txt
# 将 somelib 添加到头文件搜索路径
include_directories ( ../somelib/)
add_executable ( Hello main.cpp )
# 链接到 Foo 库
target_link_libraries ( Hello Foo)
```
## 搜索源文件
使用 find(GLOB varname patterns) 可以自动搜索给定目录中的文件，基于一个或多个搜索模式。注意，在下面的示例中，源文件和头文件都被添加到项目中。这对于编译项目并不需要，但在使用 IDE 时非常方便，因为这也会将头文件添加到项目中。IDE可以使用头文件给出更智能的提示和跳转。

```cmake
# CMakeLists.txt
cmake_minimum_required (VERSION 2.9)
# 设置项目名称
project ( HelloProject )
file(GLOB sourcefiles "src/*.hpp" "src/*.cpp")
add_executable ( Hello ${sourcefiles} )
```

## 运行时资源

运行时资源（如 DLL、游戏资产和文本文件）通常根据相对于可执行文件的路径来读取的。

一种解决方案是将资源复制到与可执行文件相同的目录中。示例：
```
CMakeLists.txt 
someexe/main.cpp 
someexe/res.txt
```
在这个项目中，源文件假设资源位于与可执行文件相同的目录中：
```c++
// main.cpp
#include <iostream>
#include <fstream>
int main (){    
  std::fstream f ( "res.txt" );    
  std::cout << f.rdbuf();    
  return 0;
}
```
CMakeLists.txt 确保复制文件。
```cmake
# CMakeLists.txt
cmake_minimum_required (VERSION 2.9)
project ( HelloProject )# 设置项目名称
add_executable ( Hello someexe/main.cpp )
file(COPY someexe/res.txt DESTINATION Debug)
file(COPY someexe/res.txt DESTINATION Release)
```
注意：这种方法的一个问题是，如果你修改了原始资源，那么你需要重新运行 CMake。
## 外部库
外部库基本上有两种类型；动态链接库（DLLs）在运行时与二进制文件链接，静态链接库在编译时链接。

静态库的设置最简单。要使用一个静态库，编译器需要知道在哪里找到头文件，链接器需要知道实际库的位置。

除非外部库与项目一起分发，否则通常不可能知道它们的位置 - 因此，使用缓存变量是很常见的，CMake 用户可以更改位置。

静态库在 Windows 上的文件扩展名为 .lib，在大多数其他平台上为.a。

```cmake
# CMakeLists.txt
cmake_minimum_required (VERSION 2.9)
# 设置项目名称
project ( HelloProject )
set ( fooinclude "/usr/local/include" CACHE PATH "Location of foo header" )
set ( foolib "/usr/local/lib/foo.a" CACHE FILEPATH "Location of foo.a" )
include_directories ( ${fooinclude} )
add_executable ( Hello someexe/main.cpp )
target_link_libraries ( Hello ${foolib} )
```
动态链接库与静态链接库的工作方式类似。在 Windows 上，仍然需要在编译时链接到库，但实际的 DLL 链接发生在运行时。可执行文件需要能够在运行时链接器的搜索路径中找到 DLL 文件。如果 DLL 不是系统库，一个简单的解决方案是将 DLL 复制到可执行文件旁边。使用 DLL 通常需要平台特定的操作，CMake 支持使用内置变量 WIN32、APPLE、UNIX。

```cmake
# CMakeLists.txt
cmake_minimum_required (VERSION 2.9)
# 设置项目名称
project ( HelloProject )
set ( fooinclude "/usr/local/include" CACHE PATH "Location of foo header" )

set ( foolib "/usr/local/lib/foo.lib" CACHE FILEPATH "Location of foo.lib" )
set ( foodll "/usr/local/lib/foo.dll" CACHE FILEPATH "Location of foo.dll" )

include_directories ( ${fooinclude} )
add_executable ( Hello someexe/main.cpp )
target_link_libraries ( Hello ${foolib} )

IF (WIN32)
file(COPY ${foodll} DESTINATION Debug)
file(COPY ${foodll} DESTINATION Release)
ENDIF(WIN32)
```
## 自动定位库

CMake 还包含一个特性，可以使用命令 find_package() 自动查找库（基于许多建议的位置）。然而，这个特性在 macOS 和 Linux 上效果最好。https://cmake.org/Wiki/CMake:How_To_Find_Libraries.

## C++ 版本
可以使用以下命令设置 C++ 版本：
```cmake
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)
```
## 定义预处理器符号
使用 add_definitions() 向项目添加预处理器符号。

```cmake
add_definitions(-DFOO="XXX")
add_definitions(-DBAR)
```
这将创建符号 FOO 和 BAR，可以在源代码中使用：

```c
#include <iostream>
using namespace std;
int main (){
#ifdef BAR
    cout << "Bar" << endl;
#endif
    cout << "Hello world" << FOO << endl;
    return 0;
}
```
## 链接和信息
https://cmake.org/Wiki/CMake/Language_Syntax

https://cmake.org/cmake/help/v3.0/command/set.html

由 Morten Nobel-Jørgensen 创建，mnob@itu.dk，ITU，2017
根据 MIT 许可证发布。
Latex 模板由 John Smith 创建，2015 http://johnsmith.com/
根据 MIT 许可证发布。

# 其他优秀参考资料
https://zhuanlan.zhihu.com/p/534439206
