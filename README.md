<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=180&section=header&text=C%2B%2B%20%E5%AD%90%E9%9B%86%E7%BC%96%E8%AF%91%E5%99%A8&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=%E6%AD%A3%E5%88%99%E5%88%86%E6%9E%90%20%E2%86%92%20%E9%80%92%E5%BD%92%E4%B8%8B%E9%99%8D%20%E2%86%92%20AST%20%E2%86%92%20%E8%AF%AD%E4%B9%89%E5%88%86%E6%9E%90%20%E2%86%92%20%E4%B8%89%E5%9C%B0%E5%9D%80%E7%A0%81&descAlignY=55&descAlign=50" width="100%" />
</p>

| 类别 | 技术栈 |
|------|--------|
| **语言** | Python 3.7+ |
| **解析** | 递归下降，手写 |
| **中间表示** | 三地址码（四元式） |
| **目标** | C++17 子集 |
| **依赖** | 无（纯 Python） |

## 📋 简介

纯 Python 实现的 C++ 子集编译器，完整流水线：预处理 → 词法分析 → 语法分析 → 语义分析 → 三地址码生成。支持函数、控制流、指针、数组、类型检查、宏预处理等 C++17 特性。

## 🚀 快速开始

```bash
# 完整编译流水线
python preprocess.py < input.cpp
python lexer.py < input.cpp
python parser.py < input.cpp
python semantic_analyzer.py < input.cpp
python tac_generator.py < input.cpp

# 用自带示例测试
python preprocess.py < main.cpp > preprocessed.cpp
python lexer.py < preprocessed.cpp
python parser.py < preprocessed.cpp
python semantic_analyzer.py < preprocessed.cpp
python tac_generator.py < preprocessed.cpp
```

## ✨ 功能特性

- **预处理**：宏展开、`#define`/`#ifdef`/`#include` 处理
- **词法分析**：正则分词，完整 C++ 关键字/运算符支持
- **语法分析**：递归下降解析，生成带类型的 AST
- **语义分析**：符号表、类型检查、作用域管理、常量正确性
- **代码生成**：三地址码（四元式）中间表示

## 🏗️ 编译流水线

```
源代码 (C++)
    │
    ▼
┌─────────────┐
│  预处理     │  → 宏展开、条件编译
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  词法分析   │  → 分词（关键字、运算符、字面量、标识符）
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  语法分析   │  → 递归下降 → AST
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  语义分析   │  → 符号表、类型检查、作用域
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  三地址码   │  → 四元式中间表示
└─────────────┘
```

## ❓ 常见问题

| 问题 | 回答 |
|------|------|
| **能编译真实 C++ 代码吗？** | 处理 C++17 子集，无模板/异常/RTTI，教学用途 |
| **如何添加优化？** | 输出 TAC 后实现常量折叠等 TAC→TAC 变换 |
| **如何可视化 AST？** | 使用任意 AST 可视化工具，或添加 `__repr__` 调试输出 |

## 🔗 相关项目

- [xv6 OS](/WJH-makers/xv6-riscv-riscv) — 链接器/加载器概念与运行时环境
- [RingMoE](/WJH-makers/RingMOE) — 深度学习计算图编译与大规模训练

## 🎓 课程背景

武汉大学计算机学院 · 编译原理课程设计，大三。

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=100&section=footer" width="100%" />
</p>
