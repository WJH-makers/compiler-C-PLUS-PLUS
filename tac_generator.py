# tac_generator.py
# coding=utf-8
import logging
import sys

# 确保这些模块在您的项目中可用且路径正确
from lexer import Lexer, LexerError
from parser import Parser, ParseError
from preprocess import BasicPreprocessor
from semantic_analyzer import SemanticAnalyzer, print_annotated_ast  # 用于主程序驱动

try:
    from compiler_ast import *
except ImportError as e:
    print(f"警告: 无法从 compiler_ast.py 导入 AST 节点。某些功能可能受限或出错。\n{e}", file=sys.stderr)
    ASTNode = type('ASTNode', (object,), {})

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
log = logging.getLogger(__name__)


def visit_NullPtrLiteral(node):  # 新增
    log.debug("访问 NullPtrLiteral")
    return 0  # nullptr 通常表示为地址 0


def visit_BooleanLiteral(node):  # 新增
    log.debug(f"访问 BooleanLiteral: {node.value}")
    return 1 if node.value else 0  # C++ 中 bool 通常用 1/0


def visit_CharLiteral(node):
    log.debug(f"访问 CharLiteral: {repr(node.value)}")
    return repr(node.value)  # 返回带引号的字符表示


def visit_StringLiteral(node):
    log.debug(f"访问 StringLiteral: {repr(node.value)}")
    return repr(node.value)  # 返回带引号的字符串表示


def visit_FloatLiteral(node):
    log.debug(f"访问 FloatLiteral: {node.value}")
    return node.value


def visit_IntegerLiteral(node):
    log.debug(f"访问 IntegerLiteral: {node.value}")
    return node.value


def visit_Identifier(node):
    log.debug(f"访问 Identifier: {node.name}")
    return node.name


class TACGenerator:
    def __init__(self):
        self.instructions = []  # 存储生成的TAC指令列表
        self.temp_count = 0  # 用于生成唯一临时变量名的计数器
        self.label_count = 0  # 用于生成唯一标签名的计数器
        self.loop_context_stack = []  # 用于 break/continue: [(continue_label, break_label), ...]

    def _new_temp(self):
        """生成一个新的、唯一的临时变量名 (例如: _t0, _t1)。"""
        temp_name = f"_t{self.temp_count}"
        self.temp_count += 1
        return temp_name

    def _new_label(self, hint="L"):
        """生成一个新的、唯一的标签名 (例如: _L0, _L1)。"""
        label_name = f"_{hint}{self.label_count}"
        self.label_count += 1
        return label_name

    def _emit(self, *args):
        """将一条 TAC 指令 (表示为元组) 添加到指令列表中。"""
        self.instructions.append(args)
        log.debug(f"生成 TAC: {args}")

    # --- 核心生成方法 ---
    def generate(self, node):
        """
        公开的入口方法，用于启动 TAC 生成过程。
        Args:
            node: AST 的根节点。
        Returns:
            生成的 TAC 指令列表。
        """
        log.info("开始三地址码生成...")
        self.instructions = []
        self.temp_count = 0
        self.label_count = 0
        self.loop_context_stack = []  # 重置循环上下文
        self.visit(node)  # 开始遍历 AST
        log.info(f"三地址码生成完成，共 {len(self.instructions)} 条指令。")
        return self.instructions

    # --- AST 遍历方法 (Visitor Pattern) ---
    def visit(self, node):
        """
        通用的 visit 方法，根据节点类型分派到具体的 visit_NodeType 方法。
        """
        if node is None:
            return None

        if isinstance(node, (int, float)):  # bool 会被 int 处理
            return node
        if isinstance(node, str):
            return node

        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        log.debug(f"访问节点: {type(node).__name__} (位于 L{getattr(node, 'line', '?')})")
        try:
            result = visitor(node)
            return result
        except Exception as e:
            log.error(f"访问 {type(node).__name__} 节点时出错 (位于 L{getattr(node, 'line', '?')}): {e}", exc_info=True)
            return None  # 指示此节点的TAC生成失败

    def generic_visit(self, node):
        """处理没有特定 visit 方法的 AST 节点类型。"""
        node_type = type(node).__name__
        log.warning(f"没有为 {node_type} 节点类型实现特定的 TAC visit 方法。将尝试访问其子节点。")
        for attr_name in dir(node):
            if not attr_name.startswith('_'):
                try:
                    value = getattr(node, attr_name)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ASTNode):  # 确保是ASTNode的实例
                                self.visit(item)
                    elif isinstance(value, ASTNode):  # 确保是ASTNode的实例
                        self.visit(value)
                except AttributeError:
                    pass
                except Exception as e:
                    log.error(f"访问 {node_type} 的子节点 {attr_name} 时出错: {e}", exc_info=False)
        return None

    def visit_Program(self, node):
        log.debug("访问 Program")
        for decl in node.declarations:
            self.visit(decl)

    def visit_FunctionDefinition(self, node):
        func_name = node.name.name
        log.debug(f"访问函数定义: {func_name}")
        self._emit('FUNC_BEGIN', func_name, len(node.params))
        for i, param in enumerate(node.params):
            if param.name:  # 参数可能有名字也可能没有
                self._emit('PARAM_VAR', i, param.name.name)
        self.visit(node.body)
        self._emit('FUNC_END', func_name)

    def visit_CompoundStatement(self, node):
        log.debug("访问 CompoundStatement")
        for stmt in node.statements:
            self.visit(stmt)

    def visit_DeclarationStatement(self, node):
        var_name = node.name.name
        log.debug(f"访问 DeclarationStatement: {var_name} (类型: {node.decl_type})")
        if node.initializer:
            init_addr = self.visit(node.initializer)
            if init_addr is not None:
                self._emit('=', var_name, init_addr)
            else:
                log.warning(f"变量 '{var_name}' 的初始化表达式未生成有效地址。")

    def visit_ExpressionStatement(self, node):
        log.debug("访问 ExpressionStatement")
        self.visit(node.expression)

    def visit_IfStatement(self, node):
        log.debug("访问 IfStatement")
        else_label = self._new_label("ELSE")
        end_label = self._new_label("END_IF")
        target_label_if_false = else_label if node.else_branch else end_label

        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("If 语句的条件表达式未生成有效地址。")
            return

        self._emit('ifFalsegoto', cond_addr, target_label_if_false)
        self.visit(node.then_branch)

        if node.else_branch:
            self._emit('goto', end_label)
            self._emit('LABEL', else_label)
            self.visit(node.else_branch)
        self._emit('LABEL', end_label if node.else_branch else target_label_if_false)

    def visit_WhileStatement(self, node):
        log.debug("访问 WhileStatement")
        start_label = self._new_label("WHILE_START")
        body_label = self._new_label("WHILE_BODY")  # 可选，用于清晰
        end_label = self._new_label("WHILE_END")

        self.loop_context_stack.append({'continue': start_label, 'break': end_label})  # continue 回到条件判断
        log.debug(f"  压入循环上下文 (While): continue_label={start_label}, break_label={end_label}")

        self._emit('LABEL', start_label)
        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("While 语句的条件表达式未生成有效地址。")
            self.loop_context_stack.pop()  # 确保即使出错也弹出上下文
            return

        self._emit('ifFalsegoto', cond_addr, end_label)
        # self._emit('LABEL', body_label) # 如果需要明确的循环体标签
        self.visit(node.body)
        self._emit('goto', start_label)
        self._emit('LABEL', end_label)

        popped_context = self.loop_context_stack.pop()
        log.debug(f"  弹出循环上下文 (While): {popped_context}")

    def visit_ForStatement(self, node):
        log.debug("访问 ForStatement")

        # 1. 初始化 (Initialization)
        if node.init:
            log.debug("  处理 ForStatement.init")
            self.visit(node.init)

        # 2. 创建循环标签
        cond_label = self._new_label("FOR_COND")
        update_label = self._new_label("FOR_UPDATE")
        end_label = self._new_label("FOR_END")

        self.loop_context_stack.append({'continue': update_label, 'break': end_label})
        log.debug(f"  压入循环上下文 (For): continue_label={update_label}, break_label={end_label}")

        # 3. 放置条件标签
        self._emit('LABEL', cond_label)
        log.debug(f"  放置条件标签: {cond_label}")

        # 4. 条件判断 (Condition)
        if node.condition:
            log.debug("  处理 ForStatement.condition")
            cond_addr = self.visit(node.condition)
            if cond_addr is None:
                log.error("For 语句的条件表达式未生成有效地址。假定为假并跳出。")
                self._emit('goto', end_label)
            else:
                self._emit('ifFalsegoto', cond_addr, end_label)
                log.debug(f"  生成条件跳转: ifFalsegoto {cond_addr}, {end_label}")
        else:
            # No condition means an infinite loop (broken by break or return)
            log.debug("  ForStatement.condition 为空，视为真。")
            pass  # Continues to body

        # 5. 循环体 (Body)
        log.debug("  处理 ForStatement.body")
        self.visit(node.body)  # Body is visited before update

        # 6. 放置更新语句标签 (continue 跳转到这里)
        self._emit('LABEL', update_label)
        log.debug(f"  放置更新标签: {update_label}")

        # 7. 更新 (Update)
        if node.update:
            log.debug("  处理 ForStatement.update")
            self.visit(node.update)

        # 8. 跳转回条件判断
        self._emit('goto', cond_label)
        log.debug(f"  生成无条件跳转: goto {cond_label}")

        # 9. 放置循环结束标签
        self._emit('LABEL', end_label)
        log.debug(f"  放置结束标签: {end_label}")

        popped_context = self.loop_context_stack.pop()
        log.debug(f"  弹出循环上下文 (For): {popped_context}")
        return None

    def visit_DoWhileStatement(self, node):
        log.debug("访问 DoWhileStatement")
        start_label = self._new_label("DO_WHILE_START")
        # cond_label = self._new_label("DO_WHILE_COND") # 条件判断标签
        end_label = self._new_label("DO_WHILE_END")  # 循环结束标签

        self.loop_context_stack.append({'continue': start_label, 'break': end_label})  # continue 跳到条件判断处
        log.debug(f"  压入循环上下文 (DoWhile): continue_label={start_label}, break_label={end_label}")

        self._emit('LABEL', start_label)
        self.visit(node.body)

        # self._emit('LABEL', cond_label) # 放置条件标签
        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("Do-While 语句的条件表达式未生成有效地址。")
            self.loop_context_stack.pop()
            return

        self._emit('ifTruegoto', cond_addr, start_label)  # 如果条件为真，跳回循环开始
        self._emit('LABEL', end_label)  # 循环结束

        popped_context = self.loop_context_stack.pop()
        log.debug(f"  弹出循环上下文 (DoWhile): {popped_context}")

    def visit_ReturnStatement(self, node):
        log.debug("访问 ReturnStatement")
        if node.value:
            value_addr = self.visit(node.value)
            if value_addr is not None:
                self._emit('RETURN', value_addr)
            else:
                log.error("Return 语句的表达式未生成有效地址。")
                self._emit('RETURN')  # 或者不发出指令，取决于策略
        else:
            self._emit('RETURN')

    def visit_BreakStatement(self, node):
        log.debug("访问 BreakStatement")
        if not self.loop_context_stack:
            log.error("BreakStatement 不在循环内 (TAC 上下文栈为空)。")
            # 语义分析阶段应已捕获此错误
            return None
        break_label = self.loop_context_stack[-1]['break']
        self._emit('goto', break_label)
        log.debug(f"  生成 Break: goto {break_label}")
        return None

    def visit_ContinueStatement(self, node):
        log.debug("访问 ContinueStatement")
        if not self.loop_context_stack:
            log.error("ContinueStatement 不在循环内 (TAC 上下文栈为空)。")
            # 语义分析阶段应已捕获此错误
            return None
        continue_label = self.loop_context_stack[-1]['continue']
        self._emit('goto', continue_label)
        log.debug(f"  生成 Continue: goto {continue_label}")
        return None

    def visit_BinaryOp(self, node):
        op = node.op
        log.debug(f"访问 BinaryOp: {op} (L {node.left}, R {node.right})")

        if op == '=':
            # 左值必须是标识符、数组下标或成员访问 (简化处理)
            # 更完整的左值处理需要知道左侧的地址
            if isinstance(node.left, Identifier):
                var_name = node.left.name  # 或者 self.visit(node.left)
                rhs_addr = self.visit(node.right)
                if rhs_addr is not None:
                    self._emit('=', var_name, rhs_addr)
                else:
                    log.error(f"赋值运算符 '=' 的右侧表达式未生成有效地址。")
                return None  # 赋值语句不返回值地址 (C中会返回，但TAC中通常不直接用)
            elif isinstance(node.left, (ArraySubscript, MemberAccess)):
                # 访问左侧以获取其“地址”表示（可能是一个计算结果）
                lvalue_addr_components = self.visit_lvalue(node.left)  # 需要一个专门的 lvalue 访问方法
                rhs_addr = self.visit(node.right)
                if lvalue_addr_components and rhs_addr is not None:
                    base, index_or_member = lvalue_addr_components
                    if isinstance(node.left, ArraySubscript):
                        self._emit('=[]', base, index_or_member, rhs_addr)
                    elif isinstance(node.left, MemberAccess):
                        # 假设成员访问('.=') 或指针成员访问('->=')
                        op_store = '.=' if not node.left.is_pointer_access else '->='
                        self._emit(op_store, base, index_or_member, rhs_addr)

                else:
                    log.error(f"复杂左值 '{type(node.left).__name__}' 或右侧表达式的TAC生成失败。")
                return None  # 赋值语句
            else:
                log.error(f"复杂左值 '{type(node.left).__name__}' 的赋值 TAC 尚未完全实现。")
                return None
        elif op in ['+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']:
            if not isinstance(node.left, Identifier):  # 简化：只处理简单变量的复合赋值
                log.error(f"复合赋值运算符 '{op}' 的左操作数必须是简单标识符 (当前 TAC 限制)。")
                # 更完整的实现会像普通赋值一样处理复杂左值
                return None
            lhs_addr = self.visit(node.left)  # 得到变量名
            rhs_addr = self.visit(node.right)
            if lhs_addr is None or rhs_addr is None:
                log.error(f"复合赋值运算符 '{op}' 的操作数未能生成有效地址。")
                return None

            base_op = op[:-1]
            self._emit(base_op, lhs_addr, lhs_addr, rhs_addr)  # x = x + y (如果指令可以直接修改第一个操作数)
            # 或者用临时变量: t = x+y; x=t
            return lhs_addr  # 复合赋值表达式的值是赋回后的值
        else:  # 其他二元运算符
            lhs_addr = self.visit(node.left)
            rhs_addr = self.visit(node.right)
            if lhs_addr is None or rhs_addr is None:
                log.error(f"二元运算符 '{op}' 的操作数未能生成有效地址。")
                return None
            dest_addr = self._new_temp()
            self._emit(op, dest_addr, lhs_addr, rhs_addr)
            return dest_addr

    def visit_UnaryOp(self, node):
        op = node.op
        log.debug(f"访问 UnaryOp: {op}")
        operand_addr = self.visit(node.operand)
        if operand_addr is None:
            log.error(f"一元运算符 '{op}' 的操作数未生成有效地址。")
            return None

        dest_addr = self._new_temp()
        if op in ['+', '-', '!', '~', '&', '*']:  # '&' (取地址), '*' (解引用)
            # 一元 '+' 通常无操作，可以优化掉
            if op == '+':
                return operand_addr  # 直接返回操作数地址

            self._emit(op, dest_addr, operand_addr)
        elif op in ['p++', 'p--']:  # 后缀
            # 语义: t = x; x = x +/- 1; return t;
            self._emit('=', dest_addr, operand_addr)  # dest_addr (t) = operand_addr (x)
            base_op = '+' if op == 'p++' else '-'
            # 假设操作数是变量名，可以直接修改
            if isinstance(node.operand, Identifier):
                self._emit(base_op, node.operand.name, node.operand.name, 1)  # x = x +/- 1
            else:
                # 如果操作数不是简单标识符，则需要更复杂的左值处理
                log.error(f"后缀 {op} 的操作数不是简单标识符，TAC生成复杂。")
                # t_new = operand_addr +/- 1; store t_new to LVALUE of operand_addr
                # 这是一个简化，实际需要分解左值
            return dest_addr  # 返回原始值
        elif op in ['++p', '--p']:  # 前缀
            # 语义: x = x +/- 1; return x;
            base_op = '+' if op == '++p' else '-'
            if isinstance(node.operand, Identifier):
                self._emit(base_op, node.operand.name, node.operand.name, 1)  # x = x +/- 1
                self._emit('=', dest_addr, node.operand.name)  # dest_addr = x (new value)
            else:
                log.error(f"前缀 {op} 的操作数不是简单标识符，TAC生成复杂。")
                # LVALUE of operand_addr = operand_addr +/- 1; dest_addr = LVALUE of operand_addr
            return dest_addr  # 返回新值
        elif op == 'sizeof':
            self._emit('sizeof', dest_addr, operand_addr)

        else:
            log.warning(f"未知一元运算符 '{op}'。")
            return None
        return dest_addr

    def visit_CallExpression(self, node):
        log.debug("访问 CallExpression")
        arg_addrs = []
        for arg in node.args:
            addr = self.visit(arg)
            if addr is None:
                log.error(f"函数调用的参数 {arg} 未能生成有效地址。")
                return None
            arg_addrs.append(addr)

        for addr in arg_addrs:
            self._emit('PARAM', addr)

        if isinstance(node.function, Identifier):
            func_name = node.function.name
            result_addr = self._new_temp()  # 假设所有函数都有返回值的地方
            self._emit('CALL', result_addr, func_name, len(node.args))
            return result_addr
        else:
            # 函数指针调用
            func_ptr_addr = self.visit(node.function)
            if func_ptr_addr is None:
                log.error("函数指针表达式未能生成有效地址。")
                return None
            result_addr = self._new_temp()
            self._emit('CALL_PTR', result_addr, func_ptr_addr, len(node.args))
            return result_addr

    def visit_CastExpression(self, node):
        log.debug(f"访问 CastExpression (to type '{node.target_type}')")
        expr_addr = self.visit(node.expression)
        if expr_addr is None:
            log.error("类型转换的表达式未生成有效地址。")
            return None
        dest_addr = self._new_temp()
        # ('CAST', dest, target_type_str, source_addr)
        self._emit('CAST', dest_addr, node.target_type, expr_addr)
        return dest_addr

    def visit_TernaryOp(self, node):  # 新增
        log.debug("访问 TernaryOp")
        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("三元运算符的条件部分未生成有效地址。")
            return None

        true_label = self._new_label("TERNARY_TRUE")
        end_label = self._new_label("TERNARY_END")
        result_addr = self._new_temp()  # 用于存储最终结果

        self._emit('ifTruegoto', cond_addr, true_label)  # 如果条件为真，跳到true部分

        # False part
        log.debug("  处理 TernaryOp.false_expression")
        false_val_addr = self.visit(node.false_expression)
        if false_val_addr is not None:
            self._emit('=', result_addr, false_val_addr)
        else:
            log.error("三元运算符的 false 分支表达式未生成有效地址。")
            # 即使一个分支失败，也需要完成结构
        self._emit('goto', end_label)

        # True part
        self._emit('LABEL', true_label)
        log.debug("  处理 TernaryOp.true_expression")
        true_val_addr = self.visit(node.true_expression)
        if true_val_addr is not None:
            self._emit('=', result_addr, true_val_addr)
        else:
            log.error("三元运算符的 true 分支表达式未生成有效地址。")

        self._emit('LABEL', end_label)
        return result_addr

    # 辅助方法，用于处理数组下标和成员访问作为左值的情况
    def visit_lvalue(self, node):
        """
        专门用于访问可能作为左值的节点 (ArraySubscript, MemberAccess)。
        返回一个元组或特定结构，表示访问该左值所需的信息。
        例如:
        - ArraySubscript: (array_base_addr, index_expr_addr)
        - MemberAccess: (object_addr, member_name_or_offset)
        """
        if isinstance(node, ArraySubscript):
            log.debug("访问左值 ArraySubscript")
            array_addr = self.visit(node.array_expression)  # 通常是数组名或指针
            index_addr = self.visit(node.index_expression)
            if array_addr is not None and index_addr is not None:
                return array_addr, index_addr
        elif isinstance(node, MemberAccess):
            log.debug("访问左值 MemberAccess")
            obj_addr = self.visit(node.object_or_pointer_expression)
            member_name = node.member_identifier.name  # 直接用成员名
            if obj_addr is not None:
                return obj_addr, member_name
        elif isinstance(node, Identifier):  # 简单标识符也是左值
            log.debug(f"访问左值 Identifier: {node.name}")
            return node.name, None
        else:
            log.error(f"节点类型 {type(node).__name__} 不是预期的左值类型。")
        return None

    def visit_ArraySubscript(self, node):
        log.debug("访问 ArraySubscript (作为右值)")
        # 作为右值时，表示从数组/指针加载一个值
        # ('=[]', dest_temp, array_name, index_expr) ; 这是赋值
        # ('[]', dest_temp, array_name, index_expr)  ; 这是取值
        array_addr = self.visit(node.array_expression)
        index_addr = self.visit(node.index_expression)
        if array_addr is None or index_addr is None:
            log.error("数组下标操作的操作数未能生成有效地址。")
            return None
        dest_addr = self._new_temp()
        self._emit('[]', dest_addr, array_addr, index_addr)  # 取值指令
        return dest_addr

    def visit_MemberAccess(self, node):
        log.debug(f"访问 MemberAccess (op: {'->' if node.is_pointer_access else '.'}) (作为右值)")
        obj_addr = self.visit(node.object_or_pointer_expression)
        member_name = node.member_identifier.name  # 直接使用成员名
        if obj_addr is None:
            log.error("成员访问的基础对象/指针表达式未生成有效地址。")
            return None
        dest_addr = self._new_temp()
        # 根据是指针访问还是直接访问选择不同指令
        op_fetch = '->' if node.is_pointer_access else '.'
        self._emit(op_fetch, dest_addr, obj_addr, member_name)  # 取值指令
        return dest_addr


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <input_file.cpp>", file=sys.stderr)
        sys.exit(1)

    input_file_path = sys.argv[1]
    raw_code = None
    processed_code = None
    tokens = []
    ast = None
    line_map = {}
    had_errors = False
    semantic_success = False  # 初始化

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

    # --- 阶段 2: 预处理 ---
    if not had_errors:
        print("\n--- Stage 2: Preprocessing ---")
        try:
            preprocessor = BasicPreprocessor()
            processed_code, line_map = preprocessor.process(raw_code)
            print("预处理完成。")
        except NameError:  # 如果 BasicPreprocessor 导入失败
            print("警告: 预处理器 BasicPreprocessor 未找到或导入失败。跳过此阶段。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
        except Exception as e:
            print(f"预处理期间出错: {e}。使用原始代码。", file=sys.stderr)
            log.exception("预处理器崩溃")  # 使用log记录详细异常
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}

    # --- 阶段 3: 词法分析 ---
    if not had_errors and processed_code is not None:
        print("\n--- Stage 3: Lexical Analysis ---")
        try:
            lexer = Lexer(processed_code, line_map)
            tokens = list(lexer.tokenize())
            print(f"词法分析完成。生成了 {len(tokens)} 个 Tokens。")
        except NameError:  # Lexer导入失败
            print("错误: 词法分析器 Lexer 未找到或导入失败。", file=sys.stderr)
            had_errors = True
        except LexerError as e:
            print(f"词法分析错误: {e}", file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"意外的词法分析错误: {e}", file=sys.stderr)
            log.exception("词法分析器崩溃")
            had_errors = True

    # --- 阶段 4: 语法分析 ---
    if not had_errors and tokens:  # 确保有tokens再解析
        print("\n--- Stage 4: Parsing ---")
        is_effectively_empty = (len(tokens) == 1 and tokens[0].type == 'EOF')
        if is_effectively_empty:
            print("输入在词法分析后有效为空。创建空的 Program AST。")
            eof_token = tokens[0]
            prog_line = eof_token.original_line if hasattr(eof_token, 'original_line') else 1
            prog_col = eof_token.column if hasattr(eof_token, 'column') else 1
            # 确保 Program 导入成功
            try:
                Program
            except NameError:
                Program = type('Program', (ASTNode,),
                               {'__init__': lambda s, d, line=1, column=1: setattr(s, 'declarations', d)})

            ast = Program([], line=prog_line, column=prog_col)
        else:
            try:
                parser = Parser(tokens)  # 确保 Parser 导入成功
                ast = parser.parse_program()
                print("语法分析完成。")
                if ast is None:  # 解析器可能在某些情况下返回None
                    print("错误: 解析器未能为非空输入生成 AST。", file=sys.stderr)
                    had_errors = True
            except NameError:  # Parser 导入失败
                print("错误: 解析器 Parser 未找到或导入失败。", file=sys.stderr)
                had_errors = True
            except ParseError as e:  # ParseError 是 parser.py 中定义的
                # ParseError 通常会在其 __init__ 中打印信息，这里可以只标记错误
                print(f"语法分析失败。错误: {e}", file=sys.stderr)  # 明确打印错误
                had_errors = True
            except Exception as e:
                print(f"意外的语法分析错误: {e}", file=sys.stderr)
                log.exception("解析器崩溃")
                had_errors = True
    elif not had_errors and not tokens:  # 如果预处理后代码为空，则tokens也可能为空
        print("\n词法分析器未产生Tokens (输入可能为空)。")
        # 创建一个空的 Program AST
        try:
            Program
        except NameError:
            Program = type('Program', (ASTNode,),
                           {'__init__': lambda s, d, line=1, column=1: setattr(s, 'declarations', d)})
        ast = Program([], line=1, column=1)

    # --- 阶段 5: 语义分析 ---
    if not had_errors and ast is not None:
        print("\n--- Stage 5: Semantic Analysis ---")
        try:
            analyzer = SemanticAnalyzer()  # 确保 SemanticAnalyzer 导入成功
            semantic_success = analyzer.analyze(ast)
            if semantic_success:
                print("\n--- Final Annotated AST (after Semantic Analysis) ---")
                print_annotated_ast(ast)  # 确保 print_annotated_ast 导入成功
                print("----------------------------------------------------")
            else:
                log.error("语义分析发现错误。")  # analyzer.analyze 应该已经打印了具体错误
                had_errors = True  # 标记错误
        except NameError:  # SemanticAnalyzer 或 print_annotated_ast 导入失败
            print("错误: 语义分析器 SemanticAnalyzer 或其辅助函数 未找到或导入失败。", file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"意外的语义分析错误: {e}", file=sys.stderr)
            log.exception("语义分析器崩溃")
            had_errors = True
            semantic_success = False  # 确保标记失败

    tac_instructions = None  # 初始化
    # --- 阶段 6: AST转换为三地址码 ---
    # 仅在语义分析成功后进行TAC生成
    if not had_errors and semantic_success and ast is not None:
        print("\n--- Stage 6: Three-Address Code Generation ---")
        try:
            generator = TACGenerator()
            tac_instructions = generator.generate(ast)
            print("\n--- Generated Three-Address Code ---")
            if tac_instructions:
                for i, instruction in enumerate(tac_instructions):
                    print(f"{i:03d}:  {instruction}")
            else:
                print("(TAC 生成失败或未产生指令)")  # 更明确的信息
                if not generator.instructions:  # 检查是否真的没有指令，还是生成器内部问题
                    print("  (TACGenerator.instructions 列表为空)")
            print("------------------------------------")
            if not tac_instructions and generator.instructions:  # 如果generate返回None但内部有指令
                log.warning("TACGenerator.generate() 返回 None，但内部指令列表非空。可能存在逻辑错误。")


        except NameError:  # TACGenerator 导入失败
            print("错误: 三地址码生成器 TACGenerator 未找到或导入失败。", file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"意外的三地址码生成错误: {e}", file=sys.stderr)
            log.exception("TAC 生成器崩溃")
            had_errors = True

    # --- Final Summary ---
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("编译因一个或多个阶段的错误而失败。")
        sys.exit(1)
    elif not semantic_success:  # 特别检查语义分析是否通过 (如果它没标记had_errors)
        print("编译在语义分析阶段失败。")
        sys.exit(1)
    elif ast is None:  # 如果AST就没有生成
        print("编译失败：未能生成抽象语法树。")
        sys.exit(1)
    elif tac_instructions is None:  # 如果TAC未生成（且前面都通过了）
        print("编译在三地址码生成阶段失败或未执行。")
        sys.exit(1)
    else:
        print("编译流程（包括三地址码生成）据报告已成功完成。")
        sys.exit(0)
