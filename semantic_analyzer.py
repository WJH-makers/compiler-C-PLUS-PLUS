# coding=utf-8
import logging
import sys

# --- AST Node Definitions ---
# 确保 compiler_ast.py 在同一目录或 PYTHONPATH 中
try:
    from compiler_ast import (
        ASTNode, Program, FunctionDefinition, Parameter, CompoundStatement,
        DeclarationStatement, AssignmentStatement, ExpressionStatement,
        IfStatement, WhileStatement, ForStatement, DoWhileStatement,
        BreakStatement, ContinueStatement, ReturnStatement, Identifier,
        IntegerLiteral, FloatLiteral, StringLiteral, CharLiteral,
        BinaryOp, UnaryOp, CallExpression, ArraySubscript, MemberAccess,
        CastExpression  # 确保导入 CastExpression
    )
except ImportError as e:
    print(f"严重错误：无法从 compiler_ast.py 导入 AST 节点定义。\n{e}", file=sys.stderr)
    sys.exit(1)

# --- Lexer and Parser ---
# 假设这些已正确定义和导入
try:
    from lexer import LexerError, Lexer
    from parser import ParseError, Parser  # 导入 Parser 以便运行完整流程
    # 尝试导入 preprocess 以便在主块中使用
    from preprocess import BasicPreprocessor
except ImportError as e:
    print(f"警告：无法导入 Lexer, Parser, 或 BasicPreprocessor。独立执行可能失败。\n{e}",
          file=sys.stderr)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# === 语义分析组件 ===

class SemanticError(Exception):
    """用于语义错误的自定义异常。"""

    def __init__(self, message, node):
        location = "UnknownLocation"
        # 尝试从节点获取位置信息
        if node and hasattr(node, 'line') and node.line is not None:
            location = f"L{node.line}"
            if hasattr(node, 'column') and node.column is not None:
                location += f":C{node.column}"
        super().__init__(f"Semantic Error at {location}: {message}")
        self.node = node


class Symbol:
    """表示符号表中的条目（变量、函数等）。"""

    def __init__(self, name, type, kind='variable', node=None, params=None, return_type=None):
        self.name = name  # 标识符名称
        self.type = type  # 类型字符串 (例如, 'int', 'float*', 'function returning int')
        self.kind = kind  # 'variable', 'function', 'type', etc.
        self.node = node  # 声明处的 AST 节点 (用于错误消息)
        self.params = params  # 对于函数: 参数符号列表或重载的类型字符串列表
        self.return_type = return_type  # 对于函数: 返回类型字符串

    def __str__(self):
        """用于符号表输出的字符串表示。"""
        decl_loc = f"(L{self.node.line})" if self.node and hasattr(self.node,
                                                                   'line') and self.node.line is not None else "(predefined/unknown)"
        if self.kind == 'function':
            param_str = "..."
            if isinstance(self.params, list):
                if len(self.params) > 0 and isinstance(self.params[0], Symbol):  # 标准参数列表
                    param_details = [f"{p.type}{' ' + p.name if p.name else ''}" for p in self.params]
                    param_str = ', '.join(param_details) if self.params else "void"
                elif len(self.params) > 0 and isinstance(self.params[0], list):  # 重载模拟 (参数类型列表的列表)
                    overloads = [f"({', '.join(p_types)})" for p_types in self.params]
                    param_str = " | ".join(overloads)
                else:  # 空列表或其他格式
                    param_str = "()" if not self.params else "(...)"
            else:  # 不是列表或 None
                param_str = "()"

            return f"Function: {self.name}{param_str} -> {self.return_type} {decl_loc}"
        else:  # 变量、类型等
            return f"{self.kind.capitalize()}: {self.name} (type: {self.type}) {decl_loc}"


class SymbolTable:
    """管理作用域和符号。"""

    def __init__(self):
        self.scopes = [{}]  # 字典堆栈，全局作用域在前
        self.current_function_symbol = None  # 跟踪当前函数以进行返回检查
        self.scope_names = ["Global"]  # 跟踪作用域名称以进行调试

    def enter_scope(self, scope_name="block"):
        """进入新作用域。"""
        parent_name = self.scope_names[-1] if self.scopes else ""
        full_scope_name = f"{parent_name}::{scope_name}"
        logging.debug(f"Entering scope: {full_scope_name} (Level {len(self.scopes)})")
        self.scopes.append({})
        self.scope_names.append(full_scope_name)

    def exit_scope(self):
        """退出当前作用域。"""
        if len(self.scopes) > 1:
            exiting_scope_name = self.scope_names.pop()
            logging.debug(f"Exiting scope: {exiting_scope_name} (Level {len(self.scopes) - 1})")
            self.scopes.pop()
        else:
            logging.error("Attempted to exit global scope!")

    def declare(self, symbol):
        """在当前作用域中声明符号，检查重复声明。"""
        current_scope = self.scopes[-1]
        name = symbol.name
        if name in current_scope:
            existing_symbol = current_scope[name]
            prev_decl_node = existing_symbol.node
            prev_decl_line = prev_decl_node.line if prev_decl_node and hasattr(prev_decl_node, 'line') else '?'
            # 抛出特定的 SemanticError
            raise SemanticError(
                f"Identifier '{name}' already declared in this scope (previous declaration at L{prev_decl_line})",
                symbol.node
            )
        scope_level = len(self.scopes) - 1
        logging.debug(
            f"Declaring '{name}' ({symbol.kind}, type: {symbol.type}) in scope Level {scope_level} ('{self.scope_names[-1]}')"
        )
        current_scope[name] = symbol
        return symbol

    def lookup(self, name, node_for_error):
        """从当前作用域向外查找符号。"""
        logging.debug(f"Looking up '{name}' from scope Level {len(self.scopes) - 1} ('{self.scope_names[-1]}')")
        for i in range(len(self.scopes) - 1, -1, -1):
            scope = self.scopes[i]
            if name in scope:
                logging.debug(f"Found '{name}' in scope Level {i} ('{self.scope_names[i]}')")
                return scope[name]

        # 在声明“未找到”之前，检查是否误用了类型名称
        if name in KNOWN_BASE_TYPES:
            raise SemanticError(f"Cannot use type name '{name}' as an identifier/variable", node_for_error)

        # 如果在任何地方都找不到
        raise SemanticError(f"Identifier '{name}' not declared", node_for_error)

    def set_current_function(self, sym):
        """设置当前正在分析的函数的符号。"""
        self.current_function_symbol = sym

    def get_current_function(self):
        """获取当前正在分析的函数的符号。"""
        return self.current_function_symbol

    def dump(self):
        """打印符号表内容以进行调试。"""
        print("\n--- Symbol Table Dump ---")
        if not self.scopes:
            print("  (Symbol table is empty)")
            return
        for i, scope in enumerate(self.scopes):
            scope_name = self.scope_names[i]
            print(f"\n-- Scope: {scope_name} (Level {i}) --")
            if not scope:
                print("  (empty)")
                continue
            for name in sorted(scope.keys()):
                print(f"  {name}: {scope[name]}")  # 使用 Symbol.__str__
        print("------------------------")


# --- 类型系统定义和辅助函数 ---
KNOWN_BASE_TYPES = {
    'void', 'int', 'float', 'char', '_Bool',  # 基本 C 类型
    'string',  # 添加的 'string' 类型
    'std::ostream', 'manipulator'  # 用于模拟 cout, endl
}


def strip_const(type_str):
    """如果存在，移除 'const ' 前缀。"""
    if isinstance(type_str, str) and type_str.startswith('const '):
        return type_str[len('const '):].strip()
    return type_str


def is_numeric(type_str):
    """检查类型字符串是否为已知的数值基础类型之一。"""
    base_type = strip_const(type_str)
    return base_type in ['int', 'float', 'char', '_Bool']


def get_pointer_base_type(type_str):
    """如果 type_str 是 'T*', 返回 'T'。否则返回 None。"""
    if isinstance(type_str, str) and type_str.endswith('*'):
        base = type_str[:-1].strip()
        return base if base else None  # Handle potential '* ' case
    return None


def make_pointer_type(base_type):
    """创建指针类型字符串，例如从 'int' 创建 'int*'。"""
    if base_type:
        # Avoid double pointers if base already ends with *? No, allow T**
        return f"{base_type}*"
    else:
        logging.warning("Attempted to create pointer type with empty base type.")
        return "* "  # 或者抛出错误？


# --- <<< MODIFIED type_compatible Function >>> ---
def type_compatible(target_type, value_type, allow_numeric_conv=True, assignment_context=False, node_for_value=None):
    """
    Checks if value_type can be implicitly converted/assigned to target_type.
    Handles identity, numeric conversion, string literal/char assignment, pointer compatibility (inc. void*, const), NULL assignment.
    """
    target_type = target_type.strip()  # Clean up types
    value_type = value_type.strip()

    if target_type == value_type:
        return True  # Identical types are compatible

    # --- Rule: Allow string -> const char* (used for function calls) ---
    if assignment_context and target_type == 'const char*' and value_type == 'string':
        logging.info(f"Compatibility rule: Allowing implicit '{value_type}' -> '{target_type}'")
        return True

    # --- Rule: Allow char*/const char*/char assignment TO string ---
    # Handles: string s = "literal"; string s += 'a'; string s += some_char_var;
    if assignment_context and target_type == 'string':
        # <<< MODIFICATION: Added 'char' to the list >>>
        if value_type in ['char*', 'const char*', 'char']:
            logging.debug(f"Allowing assignment/append: '{value_type}' -> '{target_type}'")
            return True
        # <<< END MODIFICATION >>>

    # --- Rule: Pointer compatibility ---
    target_ptr_base = get_pointer_base_type(target_type)
    value_ptr_base = get_pointer_base_type(value_type)

    if target_ptr_base is not None and value_ptr_base is not None:
        # Both are pointers
        t_base_nc = strip_const(target_ptr_base)
        v_base_nc = strip_const(value_ptr_base)
        if t_base_nc == v_base_nc or t_base_nc == 'void' or v_base_nc == 'void':
            # Check const compatibility (simplified check)
            # Cannot assign to non-const pointee from const pointee (unless target is void*)
            is_target_pointee_const = target_ptr_base.startswith('const ') or strip_const(target_ptr_base).startswith(
                'const ')
            is_value_pointee_const = value_ptr_base.startswith('const ') or strip_const(value_ptr_base).startswith(
                'const ')
            if not is_target_pointee_const and is_value_pointee_const and t_base_nc != 'void':
                logging.debug(f"Const mismatch: Cannot assign '{value_type}' to '{target_type}' (loses const pointee)")
                return False
            logging.debug(f"Pointer types compatible: '{value_type}' -> '{target_type}'")
            return True
        else:
            logging.debug(f"Pointer base type mismatch: '{v_base_nc}' vs '{t_base_nc}'")
            return False

    # --- Rule: NULL (0 literal) assignment to pointer ---
    if assignment_context and target_ptr_base is not None and value_type == 'int':
        is_literal_zero = isinstance(node_for_value, IntegerLiteral) and node_for_value.value == 0
        if is_literal_zero:
            logging.debug(f"Allowing NULL (0) assignment to pointer type '{target_type}'")
            return True

    # --- Rule: Numeric conversion (int, float, char, _Bool) ---
    if allow_numeric_conv and is_numeric(target_type) and is_numeric(value_type):
        logging.debug(f"Allowing numeric conversion: '{value_type}' -> '{target_type}'")
        return True

    # --- Default: Not compatible ---
    logging.debug(f"Types incompatible by default: '{value_type}' cannot convert/assign to '{target_type}'")
    return False


# --- <<< END MODIFIED type_compatible Function >>> ---


# === 语义分析器类 ===

class SemanticAnalyzer:
    """
    对 Parser 生成的 AST 执行语义分析。
    - 构建并使用符号表来跟踪声明和作用域。
    - 对表达式、赋值、函数调用等执行类型检查。
    - 用确定的语义类型（例如, 'int', 'float*', 'error_type'）注释 AST 节点。
    - 检测语义错误，如未声明的变量、类型不匹配等。
    """

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []  # 存储 SemanticError 消息的列表
        self.loop_depth = 0  # 跟踪循环嵌套以用于 break/continue
        self._predefine_std_symbols()  # 添加内置符号，如 cout, to_string

    def _predefine_std_symbols(self):
        """将常见的标准库符号添加到全局作用域。"""
        logging.info("Predefining standard library symbols...")
        global_scope = self.symbol_table.scopes[0]
        # 模拟 cout 和 endl
        global_scope['cout'] = Symbol(name='cout', type='std::ostream', kind='variable', node=None)
        global_scope['endl'] = Symbol(name='endl', type='manipulator', kind='variable', node=None)
        # 模拟 std::to_string 重载
        # 注意：您 C++ 代码中的 to_string 只接受 int。这里模拟标准库行为。
        global_scope['to_string'] = Symbol(
            name='to_string', type='function overload', kind='function',
            params=[['int'], ['float']],  # 参数类型列表的列表
            return_type='string', node=None
        )
        # 添加您自定义的 printMessage
        # 需要知道其参数类型和返回类型
        printMessage_params = [Symbol('msg', 'const char*', kind='parameter')]  # 创建参数符号
        global_scope['printMessage'] = Symbol(
            name='printMessage', type='function(const char*) returning void', kind='function',
            params=printMessage_params, return_type='void', node=None  # 提供参数符号列表
        )

        logging.info(f"Predefined symbols: {list(global_scope.keys())}")

    def error(self, message, node):
        """记录并记录语义错误，避免重复。"""
        try:
            err = SemanticError(message, node)
            err_str = str(err)
            if err_str not in self.errors:
                self.errors.append(err_str)
                logging.error(err_str)  # 记录错误消息
        except Exception as e:
            # 如果错误创建失败，则回退
            fallback_msg = f"Internal Error creating SemanticError: {e} (Original: {message})"
            if fallback_msg not in self.errors:
                self.errors.append(fallback_msg)
                logging.error(fallback_msg)

    def visit(self, node):
        """通用访问方法，分派到特定的 visit_NodeType 方法。"""
        if node is None:
            return None
        method_name = 'visit_' + type(node).__name__
        # 获取访问者方法，如果特定方法不存在，则回退到 generic_visit
        visitor = getattr(self, method_name, self.generic_visit)
        node_line = getattr(node, 'line', '?')
        logging.debug(f"Visiting {type(node).__name__} at L{node_line}")
        try:
            # 调用特定的访问者方法
            node_type = visitor(node)
            # 用确定的语义类型注释节点
            # 仅当 node_type 有意义时才注释（对于语句，它可能是 None）
            if node_type is not None and isinstance(node, ASTNode):
                # 基本类型注释
                setattr(node, 'semantic_type', node_type)
                logging.debug(f" -> Annotated {type(node).__name__} with type: {node_type}")
            elif node_type is None:
                # 检查是否是应该有类型的节点（表达式）返回了 None
                is_expression_node = isinstance(node, (Identifier, IntegerLiteral, FloatLiteral,
                                                       StringLiteral, CharLiteral, BinaryOp, UnaryOp,
                                                       CallExpression, ArraySubscript, MemberAccess,
                                                       CastExpression))
                if is_expression_node:
                    # 这通常是一个错误，除非显式处理
                    logging.warning(f"Visitor for expression node {type(node).__name__} returned None.")
                    # 应该返回 'error_type' 以指示问题
                    setattr(node, 'semantic_type', 'error_type')
                    return 'error_type'

            return node_type  # 返回计算出的类型或 None
        except SemanticError as se:
            # 错误已由 self.error 在访问者内部记录
            setattr(node, 'semantic_type', 'error_type')  # 将节点标记为具有错误类型
            return 'error_type'
        except Exception as e:
            # 捕获访问特定节点期间的意外错误
            self.error(f"Internal error visiting {type(node).__name__}: {e}", node)
            logging.exception(f"Unexpected exception visiting {type(node).__name__}")  # 记录堆栈跟踪
            setattr(node, 'semantic_type', 'error_type')
            return 'error_type'

    def generic_visit(self, node):
        """未处理的 AST 节点类型的回退访问者。"""
        node_name = type(node).__name__
        logging.warning(f"No specific visitor found for AST node type: {node_name}. Skipping.")
        # 如果是表达式节点，这可能导致错误
        is_expression_node = isinstance(node, (Identifier, IntegerLiteral, FloatLiteral,
                                               StringLiteral, CharLiteral, BinaryOp, UnaryOp,
                                               CallExpression, ArraySubscript, MemberAccess,
                                               CastExpression))
        if is_expression_node:
            logging.error(
                f"Generic visitor used for expression node {node_name}. This will likely cause type errors downstream.")
            return "error_type"  # 表明错误
        return None  # 对于语句等返回 None 是正常的

    # --- 访问者方法（按字母顺序排序）---

    def visit_ArraySubscript(self, node):
        """分析数组下标（例如, arr[index]）。"""
        array_expr_type = self.visit(node.array_expression)
        index_expr_type = self.visit(node.index_expression)

        if array_expr_type == "error_type" or index_expr_type == "error_type":
            return "error_type"

        # 数组表达式必须是指针类型
        base_type = get_pointer_base_type(array_expr_type)
        if base_type is None:
            self.error(f"Cannot apply subscript '[]' to non-pointer type '{array_expr_type}'", node.array_expression)
            return "error_type"
        if base_type == 'void':
            self.error("Cannot apply subscript '[]' to 'void*'", node.array_expression)
            return "error_type"

        # 索引表达式必须是整型
        if not is_numeric(index_expr_type) or strip_const(index_expr_type) == 'float':
            self.error(f"Array subscript index must be an integral type, but got '{index_expr_type}'",
                       node.index_expression)
            return "error_type"

        # 下标操作的结果是基础类型的左值
        return base_type

    def visit_AssignmentStatement(self, node):
        """分析赋值语句（当前简单：var = expr）。"""
        # 注意：此节点可能未使用，如果 '=' 被 BinaryOp 处理
        lvalue_node = node.lvalue
        lvalue_type = self.visit(lvalue_node)
        expr_type = self.visit(node.expression)

        # 检查左侧是否是有效的左值（语法上）
        is_lvalue_ok = isinstance(lvalue_node, (Identifier, ArraySubscript, MemberAccess)) or \
                       (isinstance(lvalue_node, UnaryOp) and lvalue_node.op == '*')
        if not is_lvalue_ok:
            self.error("Left-hand side of assignment is not assignable (not an lvalue)", lvalue_node)
            # 即使出错，也继续进行类型检查

        # 阻止对 const 变量的赋值（初始化在 DeclarationStatement 中处理）
        if lvalue_type.startswith('const '):
            self.error(f"Cannot assign to constant lvalue of type '{lvalue_type}'", lvalue_node)
            # 即使类型兼容，这也是一个错误

        # 检查类型兼容性
        if lvalue_type != "error_type" and expr_type != "error_type":
            if not type_compatible(lvalue_type, expr_type, assignment_context=True, node_for_value=node.expression):
                self.error(f"Cannot assign expression of type '{expr_type}' to lvalue of type '{lvalue_type}'", node)

        return None  # 赋值语句本身没有类型

    def visit_BinaryOp(self, node):
        """分析二元操作，确定结果类型。"""
        op = node.op
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)

        # 如果任一操作数类型计算出错，则传播错误
        if left_type == "error_type" or right_type == "error_type":
            return "error_type"

        # --- 操作符特定逻辑 ---

        # 赋值操作符 (=, +=, -=, etc.)
        if op in ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']:
            # 检查左操作数是否是有效的左值
            is_lvalue_ok = isinstance(node.left, (Identifier, ArraySubscript, MemberAccess)) or \
                           (isinstance(node.left, UnaryOp) and node.left.op == '*')
            if not is_lvalue_ok:
                self.error(f"Left operand of '{op}' must be an lvalue", node.left);
                return "error_type"
            # 检查左值是否为常量
            if left_type.startswith('const '):
                self.error(f"Cannot assign to constant lvalue using '{op}'", node.left)
                return "error_type"  # 赋值不允许，即使类型兼容

            # <<< 使用更新后的 type_compatible 检查类型兼容性 >>>
            if not type_compatible(left_type, right_type, assignment_context=True, node_for_value=node.right):
                self.error(f"Type mismatch for operator '{op}': Cannot assign '{right_type}' to '{left_type}'", node);
                return "error_type"

            # 对复合赋值操作符进行额外检查 (+=, -=, etc.)
            if op != '=':
                base_op = op[:-1]  # 获取基础操作符 (+, -, *)
                valid_compound = False
                # 检查数值操作数
                if is_numeric(left_type) and is_numeric(right_type):
                    # 模运算需要整型
                    if base_op == '%' and (strip_const(left_type) == 'float' or strip_const(right_type) == 'float'):
                        self.error(f"Operator '{op}' requires integral operands for modulo", node);
                        return "error_type"
                    # 位运算需要整型
                    if base_op in ['&=', '|=', '^=', '<<=', '>>='] and \
                            (strip_const(left_type) == 'float' or strip_const(right_type) == 'float'):
                        self.error(f"Operator '{op}' requires integral operands for bitwise operation", node);
                        return "error_type"
                    valid_compound = True
                # 检查指针算术 (pointer += integer, pointer -= integer)
                elif base_op in ['+', '-'] and get_pointer_base_type(left_type) is not None and strip_const(
                        right_type) == 'int':
                    if get_pointer_base_type(left_type) == 'void':
                        self.error(f"Cannot perform pointer arithmetic with '{op}' on 'void*'", node.left);
                        return "error_type"
                    valid_compound = True
                # 检查字符串连接 +=
                elif base_op == '+' and left_type == 'string' and \
                        right_type in ['string', 'char*', 'const char*', 'char']:  # << type_compatible 已经允许了，这里再次确认操作有效
                    valid_compound = True
                # (可选) 检查非标准 string += int
                # elif base_op == '+' and left_type == 'string' and right_type == 'int':
                #     logging.warning(...) valid_compound = True

                if not valid_compound:
                    # 如果 type_compatible 允许，但操作对这些类型无效（例如 float %= float）
                    self.error(
                        f"Invalid operand types ('{left_type}', '{right_type}') combination for compound assignment operator '{op}' logic",
                        node);
                    return "error_type"

            # 赋值操作的结果类型是左值的类型
            return left_type

        # 流插入操作符 (<<)
        elif op == '<<':
            # 处理流插入 (cout << ...)
            if left_type == 'std::ostream':
                # 定义允许流式传输的类型（可以扩展）
                allowed_stream_types = KNOWN_BASE_TYPES.union({'char*', 'const char*'}) - {
                    'std::ostream'}  # 移除 ostream 本身
                # 检查右侧类型是否直接允许，或者是否是指针（例如 char*），或者是否是 manipulator
                if strip_const(right_type) in allowed_stream_types or \
                        get_pointer_base_type(right_type) == 'char' or \
                        right_type == 'manipulator':
                    # 流插入的结果是流本身
                    return 'std::ostream'
                else:
                    self.error(f"Cannot apply '<<' to 'std::ostream' and operand of type '{right_type}'", node);
                    return "error_type"
            else:  # 处理位左移
                l_integral = strip_const(left_type) != 'float' and is_numeric(left_type);
                r_integral = strip_const(right_type) != 'float' and is_numeric(right_type);
                if not l_integral or not r_integral:
                    self.error(
                        f"Operands for bitwise shift '<<' must be integral, got '{left_type}' and '{right_type}'",
                        node);
                    return "error_type"
                # 结果类型通常是提升后的 int，简化为 'int'
                return 'int'

        # 算术操作符 (+, -, *, /, %)
        elif op in ['+', '-', '*', '/', '%']:
            l_ptr_base = get_pointer_base_type(left_type);
            r_ptr_base = get_pointer_base_type(right_type)
            l_num = is_numeric(left_type);
            r_num = is_numeric(right_type)
            l_str = left_type == 'string';
            r_str = right_type == 'string'

            # 指针算术: ptr + int, int + ptr, ptr - int
            if op == '+' and l_ptr_base and r_num and strip_const(right_type) == 'int':
                if l_ptr_base == 'void': self.error("Cannot perform arithmetic on 'void*'",
                                                    node.left); return "error_type"
                return left_type  # 结果是指针类型
            if op == '+' and l_num and strip_const(left_type) == 'int' and r_ptr_base:
                if r_ptr_base == 'void': self.error("Cannot perform arithmetic on 'void*'",
                                                    node.right); return "error_type"
                return right_type  # 结果是指针类型
            if op == '-' and l_ptr_base and r_num and strip_const(right_type) == 'int':
                if l_ptr_base == 'void': self.error("Cannot perform arithmetic on 'void*'",
                                                    node.left); return "error_type"
                return left_type  # 结果是指针类型

            # 指针减法: ptr - ptr
            elif op == '-' and l_ptr_base and r_ptr_base:
                # 需要指向兼容类型的指针（暂时忽略 const）
                t_base_nc = strip_const(l_ptr_base);
                v_base_nc = strip_const(r_ptr_base)
                if t_base_nc == v_base_nc and t_base_nc != 'void':
                    return 'int'  # 结果通常是 ptrdiff_t，简化为 'int'
                else:
                    # 检查是否涉及 void*，这通常不允许
                    if t_base_nc == 'void' or v_base_nc == 'void':
                        self.error(f"Cannot subtract 'void*' pointers", node);
                    else:
                        self.error(f"Cannot subtract pointers of incompatible types: '{left_type}' and '{right_type}'",
                                   node);
                    return "error_type"

            # 字符串连接: string + string, string + char*, string + char 等
            elif op == '+' and (l_str or r_str):
                other_type = right_type if l_str else left_type
                # 允许 string + (string | char* | const char* | char)
                if other_type in ['string', 'char*', 'const char*', 'char']:
                    return 'string'  # 结果是 string
                # (可选) 允许 string + int (如果你的语言支持)
                # elif other_type == 'int': return 'string'
                else:
                    self.error(f"Cannot concatenate 'string' with type '{other_type}' using '+'", node);
                    return "error_type"

            # 数值算术
            elif l_num and r_num:
                if op == '%':  # 模运算需要整型
                    l_integral = strip_const(left_type) != 'float';
                    r_integral = strip_const(right_type) != 'float';
                    if not l_integral or not r_integral:
                        self.error("Operands for modulo '%' must be integral", node);
                        return "error_type"
                    return 'int'  # % 的结果是整型
                # 确定结果类型（如果任一操作数是 float，则结果是 float）
                if strip_const(left_type) == 'float' or strip_const(right_type) == 'float':
                    return 'float'
                else:
                    return 'int'  # 如果两个操作数都是整型，则结果是 int
            else:
                # 算术运算符的操作数无效
                self.error(f"Invalid operand types ('{left_type}', '{right_type}') for operator '{op}'", node);
                return "error_type"

        # 关系和相等操作符 (==, !=, <, >, <=, >=)
        elif op in ['==', '!=', '<', '>', '<=', '>=']:
            l_ptr_base = get_pointer_base_type(left_type);
            r_ptr_base = get_pointer_base_type(right_type)
            l_num = is_numeric(left_type);
            r_num = is_numeric(right_type)
            # 检查是否与 NULL（整型字面量 0）比较
            is_left_null = l_num and isinstance(node.left, IntegerLiteral) and node.left.value == 0
            is_right_null = r_num and isinstance(node.right, IntegerLiteral) and node.right.value == 0

            can_compare = False
            # 比较两个数值类型
            if l_num and r_num:
                can_compare = True
            # 比较两个兼容类型的指针（允许 void*）
            elif l_ptr_base and r_ptr_base:
                t_base_nc = strip_const(l_ptr_base);
                v_base_nc = strip_const(r_ptr_base)
                # 允许比较相同基础类型或涉及 void* 的指针
                if t_base_nc == v_base_nc or t_base_nc == 'void' or v_base_nc == 'void':
                    can_compare = True
            # 比较指针与 NULL（0 字面量）
            elif (l_ptr_base and is_right_null) or (r_ptr_base and is_left_null):
                can_compare = True
            # (可选) 比较 string 类型 (C++ 中 string 可以用 ==, !=, < etc. 比较)
            elif left_type == 'string' and right_type == 'string':
                can_compare = True

            if not can_compare:
                self.error(f"Cannot compare operands of types '{left_type}' and '{right_type}' using operator '{op}'",
                           node);
                return "error_type"
            # 比较的结果是布尔值，映射到 'int'
            return 'int'

        # 逻辑操作符 (&&, ||)
        elif op in ['&&', '||']:
            # 操作数必须可转换为 bool（标量类型：数值或指针）
            l_scalar = is_numeric(left_type) or get_pointer_base_type(left_type) is not None
            r_scalar = is_numeric(right_type) or get_pointer_base_type(right_type) is not None
            if not l_scalar or not r_scalar:
                self.error(
                    f"Operands for logical operator '{op}' must be scalar (numeric or pointer), got '{left_type}' and '{right_type}'",
                    node);
                return "error_type"
            # 逻辑运算的结果是布尔值，映射到 'int'
            return 'int'

        # 位操作符 (&, |, ^, >>) (<< 已在上面处理)
        elif op in ['&', '|', '^', '>>']:
            l_integral = strip_const(left_type) != 'float' and is_numeric(left_type);
            r_integral = strip_const(right_type) != 'float' and is_numeric(right_type);
            if not l_integral or not r_integral:
                self.error(
                    f"Operands for bitwise operator '{op}' must be integral, got '{left_type}' and '{right_type}'",
                    node);
                return "error_type"
            # 结果类型通常是提升后的 int，简化为 'int'
            return 'int'

        else:
            # 如果解析器只产生已知的操作符，则不应到达此处
            self.error(f"Unsupported or unknown binary operator '{op}' encountered in semantic analysis", node);
            return "error_type"

    def visit_BreakStatement(self, node):
        """检查 'break' 是否在循环/switch 内部。"""
        if self.loop_depth <= 0:
            self.error("'break' statement not in loop or switch", node)
        return None  # 语句没有类型

    # --- visit_CastExpression 已添加 ---
    def visit_CastExpression(self, node):
        """分析 C 风格的类型转换表达式 (type)expr。"""
        target_type = node.target_type  # 要转换到的类型 (字符串)
        expr_type = self.visit(node.expression)  # 访问被转换的表达式

        if expr_type == "error_type":
            return "error_type"  # 传播错误

        # 基本的转换验证 (根据需要用更多 C++ 规则扩展)

        # 规则 1: 数值类型可以相互转换 (int, float, char, _Bool)
        is_target_numeric = is_numeric(target_type)
        is_expr_numeric = is_numeric(expr_type)
        if is_target_numeric and is_expr_numeric:
            logging.debug(f"Allowing numeric cast from '{expr_type}' to '{target_type}'")
            return target_type  # 返回目标类型

        # 规则 2: 整数可以转换为指针 (小心处理)
        target_is_ptr = get_pointer_base_type(target_type) is not None
        if target_is_ptr and expr_type == 'int':
            # 示例: 检查是否是将字面量 0 转换为 NULL 指针
            is_literal_zero = isinstance(node.expression, IntegerLiteral) and node.expression.value == 0
            if is_literal_zero:
                logging.debug("Cast from integer literal 0 to pointer type allowed (NULL).")
                return target_type
            else:
                # 对可能不安全的转换发出警告
                logging.warning(
                    f"Allowing potentially unsafe cast from non-zero 'int' to pointer type '{target_type}'.")
                return target_type

        # 规则 3: 指针可以转换为整数 (小心处理)
        expr_is_ptr = get_pointer_base_type(expr_type) is not None
        # 允许从指针转换为任何数值类型
        if is_numeric(target_type) and expr_is_ptr:
            logging.warning(
                f"Allowing potentially unsafe cast from pointer type '{expr_type}' to numeric type '{target_type}'.")
            return target_type

        # 规则 4: 指针类型可以相互转换 (可能不安全)
        if target_is_ptr and expr_is_ptr:
            target_base = get_pointer_base_type(target_type)
            expr_base = get_pointer_base_type(expr_type)
            is_target_pointee_const = target_base.startswith('const ')
            is_expr_pointee_const = expr_base.startswith('const ')
            if not is_target_pointee_const and is_expr_pointee_const and target_base != 'void':
                self.error(
                    f"Potentially unsafe C-style cast from '{expr_type}' to '{target_type}' may drop const qualifier from pointee",
                    node)
                # return "error_type" # Decide if this should be an error

            logging.warning(f"Allowing potentially unsafe pointer cast from '{expr_type}' to '{target_type}'.")
            return target_type  # 返回目标指针类型

        # 默认: 已实现规则不允许转换
        self.error(f"Invalid C-style cast from type '{expr_type}' to '{target_type}'", node)
        return "error_type"

    # --- <<< END visit_CastExpression >>> ---

    def visit_CallExpression(self, node):
        """分析函数调用，检查参数，返回返回类型。"""
        func_expr_node = node.function
        called_expr_type = self.visit(func_expr_node)

        if called_expr_type == "error_type": return "error_type"

        arg_types = [self.visit(arg) for arg in node.args]
        if "error_type" in arg_types: return "error_type"

        func_symbol = None
        expected_param_defs = None
        expected_return_type = "error_type"

        if isinstance(func_expr_node, Identifier):
            func_name = func_expr_node.name
            try:
                func_symbol = self.symbol_table.lookup(func_name, func_expr_node)
                if func_symbol.kind != 'function':
                    self.error(f"'{func_name}' is a {func_symbol.kind}, not a function", func_expr_node);
                    return "error_type"
                expected_param_defs = func_symbol.params
                expected_return_type = func_symbol.return_type
                # setattr(func_expr_node, 'semantic_type', func_symbol.type) # Annotate identifier if needed
            except SemanticError:
                return "error_type"

        elif called_expr_type.startswith("function"):
            logging.warning(f"Function call via expression of type '{called_expr_type}'. Argument checks limited.")
            try:
                if " returning " in called_expr_type:
                    ret_part = called_expr_type.split(" returning ")[-1]
                    expected_return_type = ret_part.strip()
                    expected_param_defs = None  # Parameter check difficult from type string alone
                else:
                    self.error(
                        f"Could not determine return type from function pointer type string: '{called_expr_type}'",
                        node);
                    return "error_type"
            except Exception as e:
                self.error(f"Error parsing function pointer type string '{called_expr_type}': {e}", node);
                return "error_type"

        else:
            self.error(f"Expression of type '{called_expr_type}' is not callable", func_expr_node);
            return "error_type"

        if expected_param_defs is not None:
            is_overload_sim = isinstance(expected_param_defs, list) and len(expected_param_defs) > 0 and isinstance(
                expected_param_defs[0], list)

            if is_overload_sim:
                match_found = False
                matched_return_type = None
                for signature in expected_param_defs:
                    if len(arg_types) == len(signature):
                        compatible = True
                        for i, (arg_t, param_t) in enumerate(zip(arg_types, signature)):
                            if not type_compatible(param_t, arg_t, assignment_context=True,
                                                   node_for_value=node.args[i]):
                                compatible = False;
                                break
                        if compatible:
                            match_found = True;
                            matched_return_type = func_symbol.return_type if func_symbol else "error_type"
                            break
                if not match_found:
                    func_name_err = func_symbol.name if func_symbol else "overloaded function"
                    arg_type_str = ', '.join(map(str, arg_types))
                    self.error(f"No matching overload for '{func_name_err}' with argument types ({arg_type_str})", node)
                    expected_return_type = "error_type"
                elif matched_return_type:
                    expected_return_type = matched_return_type

            else:  # Standard function call
                param_symbols = expected_param_defs if isinstance(expected_param_defs, list) and all(
                    isinstance(p, Symbol) for p in expected_param_defs) else []
                if len(arg_types) != len(param_symbols):
                    func_name_err = func_symbol.name if func_symbol else "function"
                    self.error(
                        f"Function '{func_name_err}' expects {len(param_symbols)} argument(s), but got {len(arg_types)}",
                        node)
                    return "error_type"
                else:
                    for i, (arg_t, param_sym) in enumerate(zip(arg_types, param_symbols)):
                        if not type_compatible(param_sym.type, arg_t, assignment_context=True,
                                               node_for_value=node.args[i]):
                            func_name_err = func_symbol.name if func_symbol else "function"
                            self.error(
                                f"Argument {i + 1} type mismatch for '{func_name_err}': Expected '{param_sym.type}' (or compatible), but got '{arg_t}'",
                                node.args[i])
                            expected_return_type = "error_type"

        return expected_return_type

    def visit_CharLiteral(self, node):
        return 'char'

    def visit_CompoundStatement(self, node):
        """分析块作用域 '{ ... }' 并访问其语句。"""
        self.symbol_table.enter_scope("Block")
        for stmt in node.statements:
            self.visit(stmt)  # 访问块内的每个语句
        self.symbol_table.exit_scope()
        return None  # 块语句本身没有类型

    def visit_ContinueStatement(self, node):
        """检查 'continue' 是否在循环内部。"""
        if self.loop_depth <= 0:
            self.error("'continue' statement not in loop", node)
        return None

    def visit_DeclarationStatement(self, node):
        """分析变量声明，添加符号，检查初始化器。"""
        var_name = node.name.name
        var_type = node.decl_type

        # 对作为声明语句解析的函数原型进行特殊处理
        if getattr(node, 'is_prototype', False):
            func_name = var_name
            return_type = var_type
            param_symbols = getattr(node, 'prototype_params', [])  # 应该是 Symbol 列表
            logging.debug(f"Processing function prototype declaration for '{func_name}'")
            try:
                existing_symbol = self.symbol_table.lookup(func_name, node.name)
                if existing_symbol.kind != 'function':
                    self.error(f"'{func_name}' previously declared as {existing_symbol.kind}", node.name)
                    return None
                # TODO: Add checks for return type and parameter compatibility here
                logging.debug(f"Prototype matches existing declaration for '{func_name}'.")
            except SemanticError as e:
                if "not declared" in str(e):
                    logging.info(f"Declaring function '{func_name}' from prototype.")
                    func_symbol = Symbol(name=func_name, kind='function', type=f"function(...) returning {return_type}",
                                         node=node, params=param_symbols, return_type=return_type)
                    try:
                        self.symbol_table.declare(func_symbol)
                    except SemanticError:
                        pass
                else:
                    pass
            return None  # 原型没有初始化器

        # --- 常规变量声明 ---
        initializer_type = None
        if node.initializer:
            initializer_type = self.visit(node.initializer)
            if initializer_type == "error_type":
                pass  # Error already logged
            elif not type_compatible(var_type, initializer_type, assignment_context=True,
                                     node_for_value=node.initializer):
                self.error(
                    f"Cannot initialize variable '{var_name}' (type '{var_type}') with expression of type '{initializer_type}'",
                    node.initializer)

        var_symbol = Symbol(var_name, var_type, kind='variable', node=node)
        try:
            self.symbol_table.declare(var_symbol)
            if node.name: setattr(node.name, 'symbol', var_symbol)
        except SemanticError:
            pass
        return None

    def visit_DoWhileStatement(self, node):
        """分析 do-while 循环。"""
        self.loop_depth += 1
        self.visit(node.body)
        cond_type = self.visit(node.condition)
        is_scalar = is_numeric(cond_type) or get_pointer_base_type(cond_type) is not None
        if cond_type != "error_type" and not is_scalar:
            self.error(f"Do-while condition must be scalar type, got '{cond_type}'", node.condition)
        self.loop_depth -= 1
        return None

    def visit_ExpressionStatement(self, node):
        """分析仅包含表达式的语句（用于副作用）。"""
        self.visit(node.expression)
        return None

    def visit_FloatLiteral(self, node):
        return 'float'

    def visit_ForStatement(self, node):
        """分析 for 循环，管理作用域。"""
        self.loop_depth += 1
        self.symbol_table.enter_scope("for_loop")
        if node.init: self.visit(node.init)
        if node.condition:
            cond_type = self.visit(node.condition)
            is_scalar = is_numeric(cond_type) or get_pointer_base_type(cond_type) is not None
            if cond_type != "error_type" and not is_scalar:
                self.error(f"For loop condition must be scalar type, got '{cond_type}'", node.condition)
        if node.update: self.visit(node.update)
        self.visit(node.body)
        self.symbol_table.exit_scope()
        self.loop_depth -= 1
        return None

    # --- visit_FunctionDefinition (Corrected Indentation) ---
    def visit_FunctionDefinition(self, node):
        """分析函数定义，添加/更新符号，处理作用域。"""
        func_name = node.name.name
        return_type = node.return_type
        param_symbols = []
        param_names = set()

        # 首先创建参数符号
        for i, param_node in enumerate(node.params):
            param_type = param_node.param_type
            param_name = param_node.name.name if param_node.name else f"__unnamed_{i}"
            if param_node.name and param_name in param_names:
                self.error(f"Duplicate parameter name '{param_name}' in function '{func_name}'", param_node.name)
            param_names.add(param_name)
            param_sym = Symbol(param_name, param_type, kind='parameter', node=param_node)
            param_symbols.append(param_sym)
            if param_node.name:
                setattr(param_node.name, 'symbol', param_sym)

        func_symbol = None
        try:
            # 检查当前作用域冲突
            current_scope_symbols = self.symbol_table.scopes[-1]
            if func_name in current_scope_symbols and current_scope_symbols[func_name].kind != 'function':
                self.error(f"'{func_name}' already declared as {current_scope_symbols[func_name].kind} in this scope",
                           node.name)
                return None

            existing_symbol = self.symbol_table.lookup(func_name, node.name)

            if existing_symbol.kind == 'function':
                is_defined = hasattr(existing_symbol.node, 'body') and existing_symbol.node.body is not None
                if is_defined and existing_symbol.node != node:
                    self.error(
                        f"Redefinition of function '{func_name}' (previous definition at L{existing_symbol.node.line})",
                        node)
                else:
                    logging.info(f"Defining function '{func_name}'.")
                func_symbol = existing_symbol
                func_symbol.node = node
                func_symbol.params = param_symbols
                func_symbol.return_type = return_type
                # TODO: Add prototype vs definition compatibility checks here
            else:
                self.error(f"'{func_name}' previously declared as non-function {existing_symbol.kind}", node.name)
                return None

        except SemanticError as e:
            if "not declared" in str(e):
                logging.info(f"First encounter of function '{func_name}', declaring and defining.")
                func_symbol = Symbol(name=func_name, kind='function', type=f"function(...) returning {return_type}",
                                     node=node, params=param_symbols, return_type=return_type)
                try:
                    self.symbol_table.declare(func_symbol)
                except SemanticError:
                    pass
            else:
                return None  # Other lookup error

        # 处理符号查找/创建失败的情况
        if func_symbol is None:
            logging.error(f"Failed to establish symbol for function '{func_name}'. Using dummy symbol.")
            func_symbol = Symbol(name=func_name, kind='function', type='error_type', node=node, params=param_symbols,
                                 return_type=return_type)

        # 注释标识符节点
        if hasattr(node, 'name') and node.name:
            setattr(node.name, 'symbol', func_symbol)
        else:
            logging.warning(f"Cannot annotate function name identifier for '{func_name}': node.name is missing.")

        # 分析函数体
        self.symbol_table.set_current_function(func_symbol)
        self.symbol_table.enter_scope(f"Function_{func_name}")
        for p_sym in param_symbols:
            try:
                self.symbol_table.declare(p_sym)
            except SemanticError:
                pass
        if node.body:
            self.visit(node.body)
        else:
            logging.warning(f"Function definition node '{func_name}' lacks a body.")
        self.symbol_table.exit_scope()
        self.symbol_table.set_current_function(None)
        return None

    # --- <<< END visit_FunctionDefinition >>> ---

    def visit_Identifier(self, node):
        """查找标识符，返回其类型。"""
        name = node.name
        try:
            symbol = self.symbol_table.lookup(name, node)
            setattr(node, 'symbol', symbol)
            return symbol.type
        except SemanticError:
            return "error_type"

    def visit_IfStatement(self, node):
        """分析 if 语句，检查条件类型。"""
        cond_type = self.visit(node.condition)
        is_scalar = is_numeric(cond_type) or get_pointer_base_type(cond_type) is not None
        if cond_type != "error_type" and not is_scalar:
            self.error(f"If condition must be scalar type, got '{cond_type}'", node.condition)
        self.visit(node.then_branch)
        if node.else_branch:
            self.visit(node.else_branch)
        return None

    def visit_IntegerLiteral(self, node):
        return 'int'

    def visit_MemberAccess(self, node):
        """分析成员访问 obj.member 或 ptr->member。"""
        object_or_ptr_type = self.visit(node.object_or_pointer_expression)
        member_name = node.member_identifier.name
        if object_or_ptr_type == "error_type": return "error_type"
        logging.warning(
            f"Member access analysis requires struct/class definitions. Skipping check for '.{member_name}' or '->{member_name}'.")
        if node.is_pointer_access:
            base_struct_type = get_pointer_base_type(object_or_ptr_type)
            if base_struct_type is None:
                self.error(f"Operator '->' requires a pointer operand, got '{object_or_ptr_type}'",
                           node.object_or_pointer_expression);
                return "error_type"
        else:
            if get_pointer_base_type(object_or_ptr_type) is not None:
                self.error(f"Operator '.' requires a non-pointer operand, got '{object_or_ptr_type}'",
                           node.object_or_pointer_expression);
                return "error_type"
        return "unknown_member_type"  # Or "error_type"

    def visit_Parameter(self, node):
        """访问参数节点（主要由 FunctionDefinition 处理）。"""
        if node.name: self.visit(node.name)
        return node.param_type

    def visit_Program(self, node):
        """分析顶层程序结构。"""
        logging.info("Starting analysis of Program node.")
        # Global scope should be implicitly present from __init__
        for decl in node.declarations:
            self.visit(decl)
        logging.info("Finished analysis of Program node.")
        return None

    def visit_ReturnStatement(self, node):
        """分析 return 语句，根据当前函数检查类型。"""
        current_func = self.symbol_table.get_current_function()
        if not current_func:
            self.error("Return statement outside of a function", node);
            return None
        expected_type = current_func.return_type
        actual_type = 'void'
        if node.value:
            actual_type = self.visit(node.value)
            if actual_type == 'error_type': return None
        if expected_type == 'void':
            if actual_type != 'void':
                self.error(f"Cannot return value of type '{actual_type}' from void function '{current_func.name}'",
                           node.value or node)
        else:
            if actual_type == 'void':
                self.error(f"Non-void function '{current_func.name}' must return a value (expected '{expected_type}')",
                           node)
            elif not type_compatible(expected_type, actual_type, assignment_context=True, node_for_value=node.value):
                self.error(
                    f"Return type mismatch in function '{current_func.name}': Cannot return '{actual_type}' from function expecting '{expected_type}'",
                    node.value or node)
        return None

    def visit_StringLiteral(self, node):
        return 'char*'  # Represent as char* (or const char*)

    def visit_UnaryOp(self, node):
        """分析一元操作。"""
        op = node.op;
        operand_node = node.operand
        if op == 'sizeof':
            if isinstance(operand_node, ASTNode):
                self.visit(operand_node)
            elif isinstance(operand_node, str):
                pass  # sizeof(type)
            else:
                self.error("Unexpected operand type for sizeof", node)
            return 'int'

        operand_type = self.visit(operand_node)
        if operand_type == "error_type": return "error_type"

        if op in ['+', '-']:
            if not is_numeric(operand_type): self.error(f"Unary '{op}' requires numeric operand, got '{operand_type}'",
                                                        node); return "error_type"
            return operand_type
        elif op == '!':
            is_scalar = is_numeric(operand_type) or get_pointer_base_type(operand_type) is not None
            if not is_scalar: self.error(f"Operator '!' requires scalar operand, got '{operand_type}'",
                                         node); return "error_type"
            return 'int'
        elif op == '~':
            is_integral = strip_const(operand_type) != 'float' and is_numeric(operand_type)
            if not is_integral: self.error(f"Operator '~' requires integral operand, got '{operand_type}'",
                                           node); return "error_type"
            return 'int'
        elif op == '&':
            is_lvalue_ok = isinstance(operand_node, (Identifier, ArraySubscript, MemberAccess)) or (
                    isinstance(operand_node, UnaryOp) and operand_node.op == '*')
            if not is_lvalue_ok: self.error("Cannot take address of non-lvalue with '&'",
                                            operand_node); return "error_type"
            return make_pointer_type(operand_type)
        elif op == '*':
            base_type = get_pointer_base_type(operand_type)
            if base_type is None: self.error(f"Cannot dereference non-pointer type '{operand_type}'",
                                             node); return "error_type"
            if base_type == 'void': self.error("Cannot dereference 'void*'", node); return "error_type"
            return base_type
        elif op in ['p++', 'p--', '++p', '--p']:
            is_lvalue_ok = isinstance(operand_node, (Identifier, ArraySubscript, MemberAccess)) or (
                    isinstance(operand_node, UnaryOp) and operand_node.op == '*')
            if not is_lvalue_ok: self.error(f"Operand of '{op}' must be a modifiable lvalue",
                                            operand_node); return "error_type"
            if operand_type.startswith('const '): self.error(f"Cannot modify constant lvalue with '{op}'",
                                                             operand_node); return "error_type"
            is_scalar = is_numeric(operand_type) or get_pointer_base_type(operand_type) is not None
            if not is_scalar: self.error(f"Operator '{op}' requires scalar operand, got '{operand_type}'",
                                         operand_node); return "error_type"
            if get_pointer_base_type(operand_type) == 'void': self.error(f"Cannot apply '{op}' to 'void*'",
                                                                         operand_node); return "error_type"
            return operand_type
        else:
            self.error(f"Unsupported unary operator '{op}'", node);
            return "error_type"

    def visit_WhileStatement(self, node):
        """分析 while 循环，检查条件类型。"""
        self.loop_depth += 1
        cond_type = self.visit(node.condition)
        is_scalar = is_numeric(cond_type) or get_pointer_base_type(cond_type) is not None
        if cond_type != "error_type" and not is_scalar:
            self.error(f"While condition must be scalar type, got '{cond_type}'", node.condition)
        self.visit(node.body)
        self.loop_depth -= 1
        return None

    # --- 分析入口点 ---
    def analyze(self, ast_root):
        """运行语义分析过程并返回成功状态。"""
        print("\n--- Stage 5: Semantic Analysis ---")
        print("Starting semantic analysis...")
        self.errors = []  # 重置此运行的错误
        self.symbol_table = SymbolTable()  # 重置符号表
        self._predefine_std_symbols()  # 添加内置符号
        self.loop_depth = 0  # 重置循环深度
        self.visit(ast_root)  # 这会注释 AST 并收集错误
        self.symbol_table.dump()  # 分析后打印符号表
        if not self.errors:
            print("\nSemantic analysis successful.")
            return True
        else:
            print(f"\nSemantic analysis finished with {len(self.errors)} error(s).")
            return False


# === 带注释的 AST 打印函数 ===
def print_annotated_ast(node, indent="", prefix="", is_last=True):
    """递归打印 AST，如果存在，则包括 'semantic_type' 注释。"""
    # (代码与上一个回复中的 print_annotated_ast 相同，此处省略以简洁)
    if node is None: return
    connector = "└── " if is_last else "├── "
    print(f"{indent}{connector}{prefix}", end="")
    line_info = f" (L{node.line})" if hasattr(node, 'line') and node.line is not None else ""
    node_name = type(node).__name__
    details = ""
    semantic_type_info = ""
    if hasattr(node, 'semantic_type') and getattr(node, 'semantic_type', None):
        semantic_type_info = f" [type: {node.semantic_type}]"
    elif isinstance(node, (Identifier, IntegerLiteral, FloatLiteral, StringLiteral, CharLiteral, BinaryOp, UnaryOp,
                           CallExpression, ArraySubscript, MemberAccess, CastExpression)):
        semantic_type_info = " [type: <unannotated>]"
    if isinstance(node, Identifier):
        details = f" name='{node.name}'";  # ... (rest of the node details logic) ...
    elif isinstance(node, IntegerLiteral):
        details = f" value={node.value}";  # ...
    elif isinstance(node, FloatLiteral):
        details = f" value={node.value}";  # ...
    elif isinstance(node, (StringLiteral, CharLiteral)):
        details = f" value={repr(node.value)}";  # ...
    elif isinstance(node, BinaryOp):
        details = f" op='{node.op}'";  # ...
    elif isinstance(node, UnaryOp):
        details = f" op='{node.op}'";  # ... (handle sizeof type) ...
    elif isinstance(node, CastExpression):
        details = f" target_type='{node.target_type}'";  # ...
    elif isinstance(node, DeclarationStatement):
        details = f" decl_type='{node.decl_type}' name='{node.name.name}'";  # ... (handle prototype) ...
    elif isinstance(node, FunctionDefinition):
        details = f" name='{node.name.name}' return_type='{node.return_type}'";  # ...
    elif isinstance(node, Parameter):
        details = f" type='{node.param_type}' name='{node.name.name if node.name else '<unnamed>'}'";  # ...
    elif isinstance(node, MemberAccess):
        op = '->' if node.is_pointer_access else '.';
        details = f" member='{node.member_identifier.name}' op='{op}'";  # ...
    print(f"{node_name}{line_info}{details}{semantic_type_info}")
    child_indent = indent + ("    " if is_last else "│   ")
    children_to_print = []
    child_attrs_map = {Program: ['declarations'], FunctionDefinition: ['params', 'body'],
                       CompoundStatement: ['statements'], DeclarationStatement: ['name', 'initializer'],
                       AssignmentStatement: ['lvalue', 'expression'], ExpressionStatement: ['expression'],
                       IfStatement: ['condition', 'then_branch', 'else_branch'], WhileStatement: ['condition', 'body'],
                       ForStatement: ['init', 'condition', 'update', 'body'], DoWhileStatement: ['body', 'condition'],
                       ReturnStatement: ['value'], BinaryOp: ['left', 'right'], UnaryOp: ['operand'],
                       CallExpression: ['function', 'args'], ArraySubscript: ['array_expression', 'index_expression'],
                       MemberAccess: ['object_or_pointer_expression', 'member_identifier'], Parameter: ['name'],
                       CastExpression: ['expression']}
    child_attrs = child_attrs_map.get(type(node), [])
    if hasattr(node, 'is_prototype') and node.is_prototype and hasattr(node, 'prototype_params'):
        if 'params' not in child_attrs: child_attrs.append('prototype_params')
    for attr_name in child_attrs:  # ... (rest of the children printing logic) ...
        child = getattr(node, attr_name, None)
        if isinstance(node, UnaryOp) and node.op == 'sizeof' and isinstance(node.operand,
                                                                            str) and attr_name == 'operand': continue
        if child is not None:  # ... (handle list vs single node) ...
            if isinstance(child, list):
                valid_items = [item for item in child if item is not None]
                if valid_items: children_to_print.append((attr_name, valid_items, True))
            elif isinstance(child, ASTNode):
                children_to_print.append((attr_name, child, False))
    num_children = len(children_to_print)
    for i, (attr_name, child_or_list, is_list) in enumerate(children_to_print):  # ... (recursive calls) ...
        is_last_child = (i == num_children - 1)
        current_prefix = f"{attr_name}: " if attr_name else ""
        if is_list:  # ... (print list header and items recursively) ...
            list_connector = "└── " if is_last_child else "├── ";
            print(f"{child_indent}{list_connector}{current_prefix}[{len(child_or_list)} item(s)]")
            list_item_indent = child_indent + ("    " if is_last_child else "│   ")
            num_items = len(child_or_list);
            for j, item in enumerate(child_or_list): print_annotated_ast(item, indent=list_item_indent,
                                                                         prefix=f"[{j}]: ",
                                                                         is_last=(j == num_items - 1))
        elif isinstance(child_or_list, ASTNode):
            print_annotated_ast(child_or_list, indent=child_indent, prefix=current_prefix, is_last=is_last_child)
        else:
            leaf_connector = "└── " if is_last_child else "├── ";
            print(
                f"{child_indent}{leaf_connector}{current_prefix}{repr(child_or_list)}")


# === 主执行块 (重构后版本) ===
if __name__ == "__main__":
    # --- 参数检查 ---
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <input_file.cpp>", file=sys.stderr)
        sys.exit(1)

    input_file_path = sys.argv[1]

    # --- 初始化变量 ---
    raw_code = None
    processed_code = None
    tokens = []  # 初始化为空列表
    ast = None
    line_map = {}
    had_errors = False
    semantic_success = False  # 初始化语义分析成功标志

    print(f"--- 开始编译流程: {input_file_path} ---")

    # --- 阶段 1: 读取文件 ---
    print("\n--- Stage 1: Reading File ---")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_code = f.read()
        print("文件读取成功。")
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file_path}' 未找到。", file=sys.stderr)
        had_errors = True
    except Exception as e:
        print(f"读取文件时发生错误: {e}", file=sys.stderr)
        had_errors = True

    # --- 阶段 2: 预处理 (可选) ---
    if not had_errors:
        print("\n--- Stage 2: Preprocessing ---")
        try:
            # 尝试导入并运行预处理器
            preprocessor = BasicPreprocessor()  # 假设 BasicPreprocessor 已导入或定义
            processed_code, line_map = preprocessor.process(raw_code)
            print("预处理完成。")
        except NameError:  # 如果 BasicPreprocessor 未定义
            print("警告: 预处理器 BasicPreprocessor 未找到或导入失败。跳过此阶段。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
        except Exception as e:
            print(f"预处理期间出错: {e}。使用原始代码。", file=sys.stderr)
            logging.exception("预处理器崩溃")  # 记录详细错误
            processed_code = raw_code  # 回退到原始代码
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
            # 根据策略决定是否将预处理错误视为致命错误
            # had_errors = True # 如果希望预处理失败则停止

    # --- 阶段 3: 词法分析 ---
    if not had_errors and processed_code is not None:
        print("\n--- Stage 3: Lexical Analysis ---")
        try:
            lexer = Lexer(processed_code, line_map)  # 假设 Lexer 已导入或定义
            tokens = list(lexer.tokenize())  # 获取 token 列表
            print(f"词法分析完成。生成了 {len(tokens)} 个 Tokens。")
        except NameError:
            print("错误: 词法分析器 Lexer 未找到或导入失败。", file=sys.stderr)
            had_errors = True
        except LexerError as e:
            print(f"词法分析错误: {e}", file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"意外的词法分析错误: {e}", file=sys.stderr)
            logging.exception("词法分析器崩溃")
            had_errors = True

    # --- 阶段 4: 语法分析 (解析) ---
    if not had_errors:
        print("\n--- Stage 4: Parsing ---")
        # 检查 Token 列表是否有效为空
        is_effectively_empty = not tokens or (len(tokens) == 1 and tokens[0].type == 'EOF')
        if is_effectively_empty:
            print("输入在词法分析后有效为空。创建空的 Program AST。")
            eof_token = tokens[0] if tokens else None
            prog_line = eof_token.original_line if eof_token else 1
            prog_col = eof_token.column if eof_token else 1
            ast = Program([], line=prog_line, column=prog_col)  # 创建空程序节点
        else:
            try:
                parser = Parser(tokens)  # 假设 Parser 已导入或定义
                ast = parser.parse_program()
                print("语法分析完成。")
                if ast is None:
                    # 解析器对于非空输入返回 None 是一个内部错误
                    print("错误: 解析器未能为非空输入生成 AST。", file=sys.stderr)
                    had_errors = True
            except NameError:
                print("错误: 解析器 Parser 未找到或导入失败。", file=sys.stderr)
                had_errors = True
            except ParseError as e:
                # 解析器内部应该已经打印了详细错误
                print(f"语法分析失败。", file=sys.stderr)
                had_errors = True
            except Exception as e:
                print(f"意外的语法分析错误: {e}", file=sys.stderr)
                logging.exception("解析器崩溃")
                had_errors = True

    # --- 阶段 5: 语义分析 ---
    if not had_errors and ast is not None:
        print("\n--- Stage 5: Semantic Analysis ---")
        try:
            analyzer = SemanticAnalyzer()  # 创建分析器实例
            semantic_success = analyzer.analyze(ast)  # 执行分析，内部会打印符号表

            # 仅在语义分析本身成功时打印带注释的 AST
            if semantic_success:
                print("\n--- Final Annotated AST ---")
                print_annotated_ast(ast)  # 假设 print_annotated_ast 已定义
                print("---------------------------")
            else:
                # 如果 analyze() 返回 False，说明存在语义错误
                had_errors = True

        except Exception as e:
            print(f"意外的语义分析错误: {e}", file=sys.stderr)
            logging.exception("语义分析器崩溃")
            had_errors = True
            semantic_success = False

    # --- 最终总结 ---
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("编译因错误而失败。")
        sys.exit(1)
    else:
        # 如果到达这里，意味着所有阶段都未将 had_errors 设为 True
        # 并且如果进行了语义分析，semantic_success 也为 True (或未进行分析)
        print("编译流程成功完成。")
        sys.exit(0)
