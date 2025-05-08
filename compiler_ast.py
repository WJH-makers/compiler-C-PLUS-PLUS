# coding=utf-8
import sys


class ASTNode:
    def __init__(self, line=None, column=None):
        # Store location info for better error messages/debugging
        self.line = line
        self.column = column


# --- Program Structure ---
class Program(ASTNode):
    def __init__(self, declarations, line=None, column=None):
        super().__init__(line, column)
        self.declarations = declarations

    def __repr__(self): return f"Program({self.declarations})"


class FunctionDefinition(ASTNode):
    def __init__(self, return_type, name, params, body, line=None, column=None):
        super().__init__(line, column)
        self.return_type = return_type  # TypeNode or string
        self.name = name  # Identifier
        self.params = params  # List of Parameter nodes
        self.body = body  # CompoundStatement

    def __repr__(
            self): return f"FunctionDefinition(type={self.return_type}, name={self.name}, params={self.params}, body={self.body})"


class Parameter(ASTNode):
    def __init__(self, param_type, name, line=None, column=None):
        super().__init__(line, column)
        self.param_type = param_type  # TypeNode or string
        self.name = name  # Identifier or None

    def __repr__(self): return f"Parameter(type={self.param_type}, name={self.name})"


class CompoundStatement(ASTNode):
    def __init__(self, statements, line=None, column=None):
        super().__init__(line, column)
        self.statements = statements

    def __repr__(self): return f"CompoundStatement({self.statements})"


class DeclarationStatement(ASTNode):
    def __init__(self, decl_type, name, initializer=None, line=None, column=None):
        super().__init__(line, column)
        self.decl_type = decl_type
        self.name = name
        self.initializer = initializer

    def __repr__(self): return f"DeclarationStatement(type={self.decl_type}, name={self.name}, init={self.initializer})"


class AssignmentStatement(ASTNode):
    def __init__(self, lvalue, expression, line=None, column=None):
        super().__init__(line, column)
        self.lvalue = lvalue
        self.expression = expression

    def __repr__(self): return f"AssignmentStatement(lvalue={self.lvalue}, expr={self.expression})"


class ExpressionStatement(ASTNode):
    def __init__(self, expression, line=None, column=None):
        super().__init__(line, column)
        self.expression = expression

    def __repr__(self): return f"ExpressionStatement({self.expression})"


class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __repr__(self): return f"IfStatement(cond={self.condition}, then={self.then_branch}, else={self.else_branch})"


class WhileStatement(ASTNode):
    def __init__(self, condition, body, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.body = body

    def __repr__(self): return f"WhileStatement(cond={self.condition}, body={self.body})"


class ForStatement(ASTNode):
    def __init__(self, init, condition, update, body, line=None, column=None):
        super().__init__(line, column)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(
            self): return f"ForStatement(init={self.init}, cond={self.condition}, update={self.update}, body={self.body})"


class DoWhileStatement(ASTNode):
    def __init__(self, body, condition, line=None, column=None):
        super().__init__(line, column)
        self.body = body
        self.condition = condition

    def __repr__(self): return f"DoWhileStatement(body={self.body}, cond={self.condition})"


class BreakStatement(ASTNode):
    def __repr__(self): return "BreakStatement"


class ContinueStatement(ASTNode):
    def __repr__(self): return "ContinueStatement"


class ReturnStatement(ASTNode):
    def __init__(self, value=None, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self): return f"ReturnStatement(value={self.value})"


class Identifier(ASTNode):
    def __init__(self, name, line=None, column=None):
        super().__init__(line, column)
        self.name = name

    def __repr__(self): return f"Identifier(name='{self.name}')"


class IntegerLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.raw_value = value  # Keep original string
        try:
            # Determine base and clean suffixes
            val_str = value.lower().rstrip('ul')
            if val_str.startswith('0x'):
                self.value = int(val_str, 16)
            elif val_str.startswith('0b'):
                self.value = int(val_str, 2)
            elif val_str.startswith('0') and len(val_str) > 1:
                self.value = int(val_str, 8)
            else:
                self.value = int(val_str, 10)
        except ValueError:
            # Fallback or error
            print(f"Warning: Could not parse integer literal '{value}'", file=sys.stderr)
            self.value = 0

    def __repr__(self):
        return f"IntegerLiteral(value={self.value})"


class FloatLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.raw_value = value
        try:
            self.value = float(value.rstrip('fFlL'))
        except ValueError:
            self.value = float('nan')

    def __repr__(self):
        return f"FloatLiteral(value={self.value})"


class StringLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self): return f"StringLiteral(value={repr(self.value)})"


class CharLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self): return f"CharLiteral(value={repr(self.value)})"


class BinaryOp(ASTNode):
    def __init__(self, op, left, right, line=None, column=None):
        super().__init__(line, column)
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self): return f"BinaryOp(op='{self.op}', left={self.left}, right={self.right})"


class UnaryOp(ASTNode):
    def __init__(self, op, operand, line=None, column=None):
        super().__init__(line, column)
        self.op = op
        self.operand = operand  # Can be expression node or string (for sizeof(type))

    def __repr__(self): return f"UnaryOp(op='{self.op}', operand={self.operand})"


class CallExpression(ASTNode):
    def __init__(self, function, args, line=None, column=None):
        super().__init__(line, column)
        self.function = function
        self.args = args

    def __repr__(self): return f"CallExpression(function={self.function}, args={self.args})"


class ArraySubscript(ASTNode):
    def __init__(self, array_expression, index_expression, line=None, column=None):
        super().__init__(line, column)
        self.array_expression = array_expression
        self.index_expression = index_expression

    def __repr__(self): return f"ArraySubscript(array={self.array_expression}, index={self.index_expression})"


class MemberAccess(ASTNode):
    def __init__(self, object_or_pointer_expression, member_identifier, is_pointer, line=None, column=None):
        super().__init__(line, column)
        self.object_or_pointer_expression = object_or_pointer_expression
        self.member_identifier = member_identifier
        self.is_pointer_access = is_pointer

    def __repr__(self):
        op = '->' if self.is_pointer_access else '.'
        return f"MemberAccess(object={self.object_or_pointer_expression}, member='{self.member_identifier.name}', op='{op}')"


class CastExpression(ASTNode):
    def __init__(self, target_type, expression, line=None, column=None):
        super().__init__(line, column)
        self.target_type = target_type
        self.expression = expression

    def __repr__(self):
        return f"CastExpression(type='{self.target_type}', expr={self.expression})"


class BooleanLiteral(ASTNode):  # 占位定义
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self): return f"BooleanLiteral(value={self.value})"


class NullPtrLiteral(ASTNode):  # 占位定义
    def __init__(self, line=None, column=None):
        super().__init__(line, column)

    def __repr__(self): return "NullPtrLiteral"


class TernaryOp(ASTNode):  # 占位定义
    def __init__(self, condition, true_expression, false_expression, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.true_expression = true_expression
        self.false_expression = false_expression

    def __repr__(
            self): return f"TernaryOp(cond={self.condition}, then={self.true_expression}, else={self.false_expression})"
