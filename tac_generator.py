# tac_generator.py
# coding=utf-8
import logging
import sys

from lexer import Lexer, LexerError
from parser import Parser, ParseError
from preprocess import BasicPreprocessor
from semantic_analyzer import SemanticAnalyzer, print_annotated_ast

# --- (保持 imports 和 虚拟 AST 类 不变) ---
try:
    from compiler_ast import (
        ASTNode, Program, FunctionDefinition, Parameter, CompoundStatement,
        DeclarationStatement, AssignmentStatement, ExpressionStatement,
        IfStatement, WhileStatement, ForStatement, DoWhileStatement,
        BreakStatement, ContinueStatement, ReturnStatement, Identifier,
        IntegerLiteral, FloatLiteral, StringLiteral, CharLiteral,
        BinaryOp, UnaryOp, CallExpression, ArraySubscript, MemberAccess,
        CastExpression
    )
except ImportError as e:
    print(f"警告: 无法导入 AST 节点。功能受限。{e}", file=sys.stderr)
    # Define dummy classes if needed... (代码同上)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
log = logging.getLogger(__name__)


class TACGenerator:
    # --- (保持 __init__, _new_temp, _new_label, _emit, generate, visit, generic_visit 不变) ---
    def __init__(self):
        self.instructions = []  # 存储生成的TAC指令列表
        self.temp_count = 0  # 用于生成唯一临时变量名的计数器
        self.label_count = 0  # 用于生成唯一标签名的计数器
        # self.current_function_name = None # (如果需要跟踪函数上下文可以取消注释)

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
        self.visit(node)  # 开始遍历 AST
        log.info(f"三地址码生成完成，共 {len(self.instructions)} 条指令。")
        return self.instructions

    # --- AST 遍历方法 (Visitor Pattern) ---
    def visit(self, node):
        """
        通用的 visit 方法，根据节点类型分派到具体的 visit_NodeType 方法。
        Args:
            node: 当前访问的 AST 节点。
        Returns:
            通常是表示节点计算结果的 "地址" (变量名、临时变量名或常量值)，
            对于语句节点或错误情况可能返回 None。
        """
        if node is None:
            return None  # 处理可选的子节点 (如 else 分支)

        # 处理简单类型 (例如，直接在 AST 中存储的字符串)
        if isinstance(node, (int, float, str)):
            return node  # 字面量/常量直接作为其自身的 "地址"

        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        log.debug(f"访问节点: {type(node).__name__}")
        try:  # 添加 try-except 块以捕获访问者内部错误
            result = visitor(node)
            return result  # 将子节点的结果向上传递
        except Exception as e:
            log.error(f"访问 {type(node).__name__} 节点时出错: {e}", exc_info=True)
            # 发生错误，返回 None 表示无法生成此节点的地址/值
            return None

    def generic_visit(self, node):
        """处理没有特定 visit 方法的 AST 节点类型。"""
        node_type = type(node).__name__
        log.warning(f"没有为 {node_type} 节点类型实现特定的 TAC visit 方法。将尝试访问其子节点。")
        # 尝试递归访问子节点 - 这只是一个基本的回退，可能不适用于所有节点
        for attr_name in dir(node):
            if not attr_name.startswith('_'):  # 忽略私有/特殊属性
                try:
                    value = getattr(node, attr_name)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ASTNode):
                                self.visit(item)
                    elif isinstance(value, ASTNode):
                        self.visit(value)
                except AttributeError:
                    pass  # 忽略获取属性时的错误
                except Exception as e:
                    log.error(f"访问 {node_type} 的子节点 {attr_name} 时出错: {e}", exc_info=False)  # 记录但不崩溃
        return None  # 无法确定此节点的 TAC 结果

    # --- (visit_Program, visit_FunctionDefinition, ... visit_ReturnStatement 不变) ---
    # ... (此处省略未改变的 visit_ 语句方法，与简化版相同) ...
    def visit_Program(self, node):
        log.debug("访问 Program")
        for decl in node.declarations:
            self.visit(decl)

    def visit_FunctionDefinition(self, node):
        """
        处理函数定义。生成函数开始/结束标记和参数处理（简化）。
        """
        func_name = node.name.name
        log.debug(f"访问函数定义: {func_name}")
        # self.current_function_name = func_name # 如有需要，可以跟踪当前函数名

        # 1. 发出函数开始标记
        #    格式: ('FUNC_BEGIN', 函数名, 参数个数)
        self._emit('FUNC_BEGIN', func_name, len(node.params))

        # 2. (简化) 假设参数可以直接使用，仅记录参数名与索引的关系
        for i, param in enumerate(node.params):
            if param.name:
                # 格式: ('PARAM_VAR', 参数索引, 参数变量名)
                self._emit('PARAM_VAR', i, param.name.name)

        # 3. 访问函数体内的语句
        self.visit(node.body)

        # 4. 发出函数结束标记 (可选地，确保总有一个结束标签)
        # self._emit('LABEL', f"{func_name}_end") # 函数末尾的标签
        self._emit('FUNC_END', func_name)
        # self.current_function_name = None

    def visit_CompoundStatement(self, node):
        """处理复合语句 (花括号内的语句块)。"""
        log.debug("访问 CompoundStatement")
        for stmt in node.statements:
            self.visit(stmt)  # 依次访问块内的每条语句

    def visit_DeclarationStatement(self, node):
        """
        处理变量声明语句。如果带初始化，则生成赋值 TAC。
        """
        var_name = node.name.name
        log.debug(f"访问 DeclarationStatement: {var_name}")

        # 如果有初始化表达式
        if node.initializer:
            # 1. 访问初始化表达式，获取其结果地址 (可能是临时变量或常量)
            init_addr = self.visit(node.initializer)
            if init_addr is not None:
                # 2. 生成赋值指令
                #    格式: ('=', 目标变量名, 源地址/值)
                self._emit('=', var_name, init_addr)
            else:
                log.warning(f"变量 '{var_name}' 的初始化表达式未生成有效地址。")
        # 如果没有初始化，则 TAC 层面通常不需要操作 (内存分配由后续阶段处理)

    def visit_ExpressionStatement(self, node):
        """处理表达式语句 (通常用于函数调用等有副作用的表达式)。"""
        log.debug("访问 ExpressionStatement")
        # 访问其包含的表达式，目的是执行其副作用 (如函数调用)
        # 表达式本身的结果 (地址) 被丢弃
        self.visit(node.expression)

    def visit_IfStatement(self, node):
        """
        处理 if (-else) 语句。生成条件判断和跳转指令。
        """
        log.debug("访问 IfStatement")
        # 1. 创建 'else' 分支 (或 if 结束) 和 'end' 分支的标签
        else_label = self._new_label("ELSE")
        end_label = self._new_label("END_IF")
        # 如果没有 else 分支，else_label 实际上就是 end_label
        target_label_if_false = else_label if node.else_branch else end_label

        # 2. 访问条件表达式，获取其结果地址
        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("If 语句的条件表达式未生成有效地址。跳过分支生成。")
            return None  # 返回 None 表示处理失败

        # 3. 生成条件跳转指令 (如果条件为假，则跳转)
        #    格式: ('ifFalsegoto', 条件地址, 跳转目标标签)
        self._emit('ifFalsegoto', cond_addr, target_label_if_false)

        # 4. 访问 'then' (if 为真时执行) 分支
        self.visit(node.then_branch)

        # 5. 如果存在 'else' 分支:
        if node.else_branch:
            # a. 在 'then' 分支末尾添加无条件跳转，跳过 'else' 分支
            #    格式: ('goto', 跳转目标标签)
            self._emit('goto', end_label)
            # b. 放置 'else' 分支的标签
            #    格式: ('LABEL', 标签名)
            self._emit('LABEL', else_label)
            # c. 访问 'else' 分支
            self.visit(node.else_branch)
            # d. 放置 'if' 语句结束的标签
            self._emit('LABEL', end_label)
        else:
            # 如果没有 'else' 分支，则 'else_label' 就是结束标签
            self._emit('LABEL', target_label_if_false)  # 也就是 end_label
        return None  # 语句节点不返回值

    def visit_WhileStatement(self, node):
        """
        处理 while 循环语句。生成循环开始/结束标签和条件跳转。
        """
        log.debug("访问 WhileStatement")
        # 1. 创建循环开始和循环结束的标签
        start_label = self._new_label("WHILE_START")
        end_label = self._new_label("WHILE_END")

        # 2. 放置循环开始标签
        self._emit('LABEL', start_label)

        # 3. 访问条件表达式，获取其结果地址
        cond_addr = self.visit(node.condition)
        if cond_addr is None:
            log.error("While 语句的条件表达式未生成有效地址。")
            # 即使条件失败，也需要生成结束标签，否则后续代码可能出错
            self._emit('LABEL', end_label)  # 确保结束标签存在
            return None  # 指示处理失败

        # 4. 生成条件跳转指令 (如果条件为假，则跳出循环到 end_label)
        #    格式: ('ifFalsegoto', 条件地址, 跳转目标标签)
        self._emit('ifFalsegoto', cond_addr, end_label)

        # 5. 访问循环体
        self.visit(node.body)

        # 6. 在循环体末尾添加无条件跳转，跳回循环开始处重新判断条件
        #    格式: ('goto', 跳转目标标签)
        self._emit('goto', start_label)

        # 7. 放置循环结束标签
        self._emit('LABEL', end_label)
        return None  # 语句节点不返回值

    def visit_ReturnStatement(self, node):
        """处理 return 语句。"""
        log.debug("访问 ReturnStatement")
        # 如果 return 带有返回值
        if node.value:
            # 1. 访问返回值表达式，获取其地址
            value_addr = self.visit(node.value)
            if value_addr is not None:
                # 2. 生成带返回值的 RETURN 指令
                #    格式: ('RETURN', 返回值地址/值)
                self._emit('RETURN', value_addr)
            else:
                log.error("Return 语句的表达式未生成有效地址。")
                # 根据需要决定是否发出无值 RETURN
                self._emit('RETURN')  # 或者不发出指令
        else:
            # 生成无返回值的 RETURN 指令 (用于 void 函数)
            self._emit('RETURN')
        return None  # 语句节点不返回值

    # --- (visit_Identifier, visit_*Literal 不变) ---
    def visit_Identifier(self, node):
        """处理标识符 (变量名)。其 "地址" 就是其名称。"""
        log.debug(f"访问 Identifier: {node.name}")
        return node.name  # 直接返回变量名

    def visit_IntegerLiteral(self, node):
        """处理整数字面量。其 "地址" 就是其值。"""
        log.debug(f"访问 IntegerLiteral: {node.value}")
        return node.value  # 返回整数值

    def visit_FloatLiteral(self, node):
        """处理浮点数字面量。其 "地址" 就是其值。"""
        log.debug(f"访问 FloatLiteral: {node.value}")
        return node.value  # 返回浮点数值

    def visit_StringLiteral(self, node):
        """处理字符串字面量。返回其表示形式（带引号）。"""
        log.debug(f"访问 StringLiteral: {repr(node.value)}")
        # 字符串的处理方式依赖于后续阶段，这里简单返回其 repr()
        return repr(node.value)

    def visit_CharLiteral(self, node):
        """处理字符字面量。返回其表示形式（带引号）。"""
        log.debug(f"访问 CharLiteral: {repr(node.value)}")
        return repr(node.value)

    def visit_BinaryOp(self, node):
        """
        处理二元运算符表达式。
        **修改:** 正确处理复合赋值，特殊处理流输出 `<<`。
        """
        op = node.op
        log.debug(f"访问 BinaryOp: {op}")

        # --- 特殊处理: 流输出 (<<) ---
        if op == '<<':
            lhs_addr = self.visit(node.left)
            rhs_addr = self.visit(node.right)
            # --- 添加详细日志 ---
            log.debug(
                f"  处理 '<<': lhs_addr='{lhs_addr}' (type: {type(lhs_addr)}), rhs_addr='{rhs_addr}' (type: {type(rhs_addr)})")

            is_ostream = lhs_addr == 'cout'
            if is_ostream:
                log.debug("  检测到流输出 ('cout' 作为左操作数)")
                return lhs_addr  # 返回 'cout'
            else:
                log.debug(f"  未检测到流输出 (lhs_addr is not 'cout'), 将其视为位移。")
                if lhs_addr is None or rhs_addr is None:
                    log.error(f"位移运算符 '<<' 的操作数未能生成有效地址。lhs={lhs_addr}, rhs={rhs_addr}")  # <--- 错误日志
                    return None
                dest_addr = self._new_temp()
                self._emit(op, dest_addr, lhs_addr, rhs_addr)
                return dest_addr
        # --- 处理赋值 (=) ---
        elif op == '=':
            # 假设左操作数是简单变量 (Identifier)
            if isinstance(node.left, Identifier):
                var_name = node.left.name
                rhs_addr = self.visit(node.right)
                if rhs_addr is not None:
                    self._emit('=', var_name, rhs_addr)
                else:
                    log.error(f"赋值运算符 '=' 的右侧表达式未生成有效地址。")
                return None  # 赋值语句不返回值地址
            else:
                log.error(f"复杂左值 '{type(node.left).__name__}' 的赋值 TAC 尚未实现。")
                return None

        # --- 修改: 处理复合赋值 (+=, -=, /=, *=, %= ...) ---
        elif op in ['+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']:
            # 1. 左操作数必须是 LValue (简化为 Identifier)
            if not isinstance(node.left, Identifier):
                log.error(f"复合赋值运算符 '{op}' 的左操作数必须是简单标识符 (TAC 限制)。")
                return None
            lhs_addr = node.left.name  # 直接使用变量名

            # 2. 计算右操作数的值
            rhs_addr = self.visit(node.right)
            if rhs_addr is None:
                log.error(f"复合赋值运算符 '{op}' 的右侧表达式未生成有效地址。")
                return None

            # 3. 计算基础运算结果存入临时变量
            temp_addr = self._new_temp()
            base_op = op[:-1]  # 获取基础运算符, e.g., '+' from '+='
            self._emit(base_op, temp_addr, lhs_addr, rhs_addr)  # _tN = lhs op rhs

            # 4. 将临时变量的结果赋回给左操作数变量
            self._emit('=', lhs_addr, temp_addr)  # lhs = _tN

            # 5. 复合赋值表达式的值是赋回后的值，返回左操作数地址
            return lhs_addr

        # --- 处理其他标准二元运算符 ---
        else:
            lhs_addr = self.visit(node.left)
            rhs_addr = self.visit(node.right)
            if lhs_addr is None or rhs_addr is None:
                log.error(f"二元运算符 '{op}' 的操作数未能生成有效地址。")
                return None
            dest_addr = self._new_temp()
            self._emit(op, dest_addr, lhs_addr, rhs_addr)
            return dest_addr

    # --- (visit_UnaryOp, visit_CallExpression 等保持不变) ---
    # ... (此处省略未改变的 visit_ 表达式方法，与上一版本相同) ...
    def visit_UnaryOp(self, node):
        """
        处理一元运算符表达式 (+, -, !, ~, ++, --)。
        注意：简化版，未实现 & 和 *。
        """
        log.debug(f"访问 UnaryOp: {node.op}")

        # 1. 处理简单一元运算符 (+, -, !, ~)
        if node.op in ['+', '-', '!', '~']:
            # a. 访问操作数，获取其地址
            operand_addr = self.visit(node.operand)
            if operand_addr is None:
                log.error(f"一元运算符 '{node.op}' 的操作数未生成有效地址。")
                return None
            # b. 创建新的临时变量存储结果
            dest_addr = self._new_temp()
            # c. 生成一元运算指令
            #    对于 '-' 可能用 'NEG' 指令，'+' 可能忽略，'!'/'~' 用自身
            #    简化：直接使用 op 字符串
            #    格式: (运算符, 目标临时变量, 操作数地址)
            if node.op == '+':  # 一元加通常无操作
                # 优化：可以直接返回操作数地址，避免不必要的赋值
                # self._emit('=', dest_addr, operand_addr)
                # return dest_addr
                return operand_addr
            else:
                op_code = 'NEG' if node.op == '-' else node.op  # 可以使用更明确的操作码
                self._emit(op_code, dest_addr, operand_addr)
                return dest_addr  # 返回结果临时变量名

        # 2. 处理后缀自增/自减 (p++, p--)
        elif node.op in ['p++', 'p--']:
            # a. 检查操作数是否为简单变量 (LValue)
            if not isinstance(node.operand, Identifier):
                log.error(f"后缀运算符 '{node.op}' 的操作数必须是简单标识符 (TAC 限制)。")
                return None
            operand_name = node.operand.name

            # b. 创建临时变量保存原始值 (这是表达式的结果)
            original_value_temp = self._new_temp()
            self._emit('=', original_value_temp, operand_name)  # _tN = var

            # c. 创建临时变量计算新值
            new_value_temp = self._new_temp()
            arith_op = '+' if node.op == 'p++' else '-'
            self._emit(arith_op, new_value_temp, original_value_temp, 1)  # _tM = _tN +/- 1

            # d. 将新值写回原变量 (副作用)
            self._emit('=', operand_name, new_value_temp)  # var = _tM

            # e. 返回包含 *原始* 值的临时变量
            return original_value_temp

        # 3. 处理前缀自增/自减 (++p, --p)
        elif node.op in ['++p', '--p']:
            # a. 检查操作数是否为简单变量 (LValue)
            if not isinstance(node.operand, Identifier):
                log.error(f"前缀运算符 '{node.op}' 的操作数必须是简单标识符 (TAC 限制)。")
                return None
            operand_name = node.operand.name

            # b. 创建临时变量计算新值
            new_value_temp = self._new_temp()
            arith_op = '+' if node.op == '++p' else '-'
            self._emit(arith_op, new_value_temp, operand_name, 1)  # _tN = var +/- 1

            # c. 将新值写回原变量 (副作用)
            self._emit('=', operand_name, new_value_temp)  # var = _tN

            # d. 返回包含 *新* 值的临时变量 (这是表达式的结果)
            return new_value_temp

        # 4. 其他未实现的一元运算符 (&, *)
        elif node.op in ['&', '*']:
            log.warning(f"地址/解引用运算符 '{node.op}' 的 TAC 生成尚未实现。")
            # 尝试访问操作数，但结果可能无意义
            self.visit(node.operand)
            return None  # 无法生成有效地址
        else:
            log.warning(f"未知一元运算符 '{node.op}' 的 TAC 生成尚未实现。")
            return None

    def visit_CallExpression(self, node):
        """处理函数调用表达式。"""
        log.debug("访问 CallExpression")

        # 1. 处理参数 (简单实现：按顺序发出 PARAM 指令)
        arg_addrs = []
        for arg in node.args:
            addr = self.visit(arg)
            if addr is None:
                log.error(f"函数调用的参数 {arg} 未能生成有效地址。")
                return None  # 如果参数失败，则调用失败
            arg_addrs.append(addr)

        # 2. 依次发出 PARAM 指令
        #    格式: ('PARAM', 参数地址/值)
        for addr in arg_addrs:
            self._emit('PARAM', addr)

        # 3. 处理被调函数 (简单实现：假设是直接标识符)
        if isinstance(node.function, Identifier):
            func_name = node.function.name
            # a. 创建临时变量以接收返回值 (即使是 void 函数也创建，方便处理)
            result_addr = self._new_temp()
            # b. 生成 CALL 指令
            #    格式: ('CALL', 结果临时变量, 函数名, 参数个数)
            self._emit('CALL', result_addr, func_name, len(node.args))
            # c. 返回结果临时变量
            #    (注意: 对于 void 函数，这个临时变量可能不会被使用)
            #    改进：可以检查函数的语义类型，如果是void则返回None
            semantic_type = getattr(node.function, 'semantic_type', '')
            if semantic_type and 'returning void' in semantic_type:
                log.debug(f"函数 {func_name} 返回 void，不返回地址。")
                return None  # void 函数调用不产生值
            else:
                return result_addr
        else:
            # TODO: 处理函数指针等复杂情况
            log.error("函数调用 TAC 仅支持直接函数名调用。")
            return None

    # --- (visit_CastExpression 和其他跳过的节点保持不变) ---
    def visit_CastExpression(self, node):
        """处理类型转换 (简化：直接访问内部表达式，忽略转换)。"""
        log.warning("CastExpression TAC 生成被跳过 (直接访问内部表达式)。")
        return self.visit(node.expression)

    # 其他未显式处理的节点将使用 generic_visit 或在此处添加简单警告
    def visit_ForStatement(self, node):
        log.warning("ForStatement TAC 生成被跳过。")
        return None

    def visit_DoWhileStatement(self, node):
        log.warning("DoWhileStatement TAC 生成被跳过。")
        return None

    def visit_BreakStatement(self, node):
        log.warning("BreakStatement TAC 生成被跳过 (需要循环上下文)。")
        return None

    def visit_ContinueStatement(self, node):
        log.warning("ContinueStatement TAC 生成被跳过 (需要循环上下文)。")
        return None

    def visit_ArraySubscript(self, node):
        log.warning("ArraySubscript TAC 生成被跳过 (需要内存地址计算)。")
        return None

    def visit_MemberAccess(self, node):
        log.warning("MemberAccess TAC 生成被跳过 (需要结构体/类偏移)。")
        return None


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
    semantic_success = False

    print(f"--- 开始编译流程: {input_file_path} ---")

    # --- 阶段 1: 读取文件
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

    # --- 阶段 2: 预处理
    if not had_errors:
        print("\n--- Stage 2: Preprocessing ---")
        try:
            preprocessor = BasicPreprocessor()
            processed_code, line_map = preprocessor.process(raw_code)
            print("预处理完成。")
        except NameError:
            print("警告: 预处理器 BasicPreprocessor 未找到或导入失败。跳过此阶段。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
        except Exception as e:
            print(f"预处理期间出错: {e}。使用原始代码。", file=sys.stderr)
            logging.exception("预处理器崩溃")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}

    # --- 阶段 3: 词法分析
    if not had_errors and processed_code is not None:
        print("\n--- Stage 3: Lexical Analysis ---")
        try:
            lexer = Lexer(processed_code, line_map)
            tokens = list(lexer.tokenize())
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

    # --- 阶段 4: 语法分析
    if not had_errors:
        print("\n--- Stage 4: Parsing ---")
        is_effectively_empty = not tokens or (len(tokens) == 1 and tokens[0].type == 'EOF')
        if is_effectively_empty:
            print("输入在词法分析后有效为空。创建空的 Program AST。")
            eof_token = tokens[0] if tokens else None
            prog_line = eof_token.original_line if eof_token else 1
            prog_col = eof_token.column if eof_token else 1
            ast = Program([], line=prog_line, column=prog_col)
        else:
            try:
                parser = Parser(tokens)
                ast = parser.parse_program()
                print("语法分析完成。")
                if ast is None:
                    print("错误: 解析器未能为非空输入生成 AST。", file=sys.stderr)
                    had_errors = True
            except NameError:
                print("错误: 解析器 Parser 未找到或导入失败。", file=sys.stderr)
                had_errors = True
            except ParseError as e:
                print("语法分析失败。", file=sys.stderr)
                had_errors = True
            except Exception as e:
                print(f"意外的语法分析错误: {e}", file=sys.stderr)
                logging.exception("解析器崩溃")
                had_errors = True

    # --- 阶段 5: 语义分析 ---
    if not had_errors and ast is not None:
        # print("\n--- Stage 5: Semantic Analysis ---") # 已在 analyze 方法内部打印
        try:
            analyzer = SemanticAnalyzer()
            semantic_success = analyzer.analyze(ast)  # analyze 方法会打印其开始和结束消息
            if semantic_success:
                print("\n--- Final Annotated AST ---")
                print_annotated_ast(ast)
                print("---------------------------")
            else:
                # analyzer.analyze 已经打印了错误数量
                had_errors = True
        except Exception as e:
            print(f"意外的语义分析错误: {e}", file=sys.stderr)
            logging.exception("语义分析器崩溃")
            had_errors = True
    tac_instructions = None
    # --- 阶段 6: AST转换为三地址码
    if semantic_success and ast is not None:  # Check if previous stages succeeded
        try:
            generator = TACGenerator()
            tac_instructions = generator.generate(ast)  # Generate TAC
            print("\n--- Generated Three-Address Code ---")
            if tac_instructions:
                for i, instruction in enumerate(tac_instructions):
                    # Simple formatting - adjust as needed
                    print(f"{i:03d}:  {instruction}")
            else:
                print("(No TAC instructions generated or generation failed)")
            print("------------------------------------")

        except ImportError as e:
            print(f"Error: Could not import TACGenerator. {e}", file=sys.stderr)
            had_errors = True  # Mark failure
        except Exception as e:
            print(f"Unexpected error during TAC generation: {e}", file=sys.stderr)
            logging.exception("TAC Generator crashed")
            had_errors = True

    # --- Final Summary ---
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("编译因错误而失败。")
        sys.exit(1)
    elif not semantic_success:
        print("编译在语义分析阶段失败。")
        sys.exit(1)
    elif tac_instructions is None:
        print("编译在 TAC 生成阶段失败或未执行。")
        sys.exit(1)
    else:
        print("编译流程成功完成（包括 TAC 生成）。")
        sys.exit(0)
