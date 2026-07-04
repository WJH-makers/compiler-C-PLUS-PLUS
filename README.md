<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=180&section=header&text=C%2B%2B%20Subset%20Compiler&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Python-based%20Compiler%20%E2%80%93%20Lexer%20%E2%86%92%20Parser%20%E2%86%92%20AST%20%E2%86%92%20Semantic%20Analysis%20%E2%86%92%20TAC&descAlignY=55&descAlign=50" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python%203-3776AB?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Target-C%2B%2B%20Subset-00599C?style=flat-square&logo=c%2B%2B" />
  <img src="https://img.shields.io/badge/Parsing-Recursive%20Descent-00A1E9?style=flat-square" />
  <img src="https://img.shields.io/badge/IR-Three%20Address%20Code-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Complete-green?style=flat-square" />
</p>

## 📋 Overview

A **C++ subset compiler** built entirely in **Python** as a university compiler design project. Implements a complete compilation pipeline: **preprocessor** → **lexer** → **parser** → **semantic analyzer** → **three-address code generator**.

Supports C++17 language features including functions, control flow, pointers, arrays, type checking, and macro preprocessing.

## ✨ Key Features

- **Preprocessor**: Macro expansion, `#define`/`#ifdef`/`#if`/`#include` handling
- **Lexer**: Regex-based tokenization with full C++ keyword/operator support
- **Parser**: Recursive descent parser producing a typed AST
- **Semantic Analyzer**: Symbol table, type checking, scope management, const correctness
- **TAC Generator**: Three-address code (quadruple) intermediate representation
- **C++17 Support**: Functions, if/while/for/do-while, ternary, casts, member access

## 🏗️ Compiler Pipeline

```
Source Code (C++)
      │
      ▼
┌─────────────┐
│ Preprocessor│  → Macro expansion, #include, conditional compilation
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Lexer    │  → Tokenization (keywords, operators, literals, identifiers)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Parser    │  → Recursive descent → Abstract Syntax Tree (AST)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Semantic   │  → Symbol table, type checking, scope resolution
│  Analyzer   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     TAC     │  → Three-address code generation
│  Generator  │
└─────────────┘
```

## 🏗️ Module Structure

| Module | File | Description |
|--------|------|-------------|
| **Preprocessor** | `preprocess.py` | Macro expansion, `#define`/`#ifdef`/`#include`, string operations |
| **Lexer** | `lexer.py` | Token types, regex scanning, escape sequence handling (326 lines) |
| **AST** | `compiler_ast.py` | AST node definitions: Program, Function, Statement, Expression (276 lines) |
| **Parser** | `parser.py` | Recursive descent with operator precedence for C++ (800+ lines) |
| **Semantic Analyzer** | `semantic_analyzer.py` | Symbol table, type checking, const correctness (1000+ lines) |
| **TAC Generator** | `tac_generator.py` | Three-address code via visitor pattern (748 lines) |

## 🚀 Quick Start

```bash
# Run full compilation pipeline
python preprocess.py < input.cpp
python lexer.py < input.cpp
python parser.py < input.cpp
python semantic_analyzer.py < input.cpp
python tac_generator.py < input.cpp
```

### Test with included sample

```bash
python preprocess.py < main.cpp > preprocessed.cpp
python lexer.py < preprocessed.cpp
python parser.py < preprocessed.cpp
python semantic_analyzer.py < preprocessed.cpp
python tac_generator.py < preprocessed.cpp
```

## 📦 Requirements

- Python 3.7+
- No external dependencies required (pure Python implementation)

## 🎓 Academic Context

This project was completed as the final project for the **Compiler Principles** course (大三) at **Wuhan University**, School of Computer Science. It demonstrates a complete understanding of compiler construction including lexical analysis, syntax analysis, semantic analysis, and intermediate code generation.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=100&section=footer" width="100%" />
</p>
