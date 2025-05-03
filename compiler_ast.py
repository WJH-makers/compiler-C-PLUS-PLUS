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
        self.return_type = return_type # TypeNode or string
        self.name = name # Identifier
        self.params = params # List of Parameter nodes
        self.body = body # CompoundStatement
    def __repr__(self): return f"FunctionDefinition(type={self.return_type}, name={self.name}, params={self.params}, body={self.body})"

class Parameter(ASTNode):
    def __init__(self, param_type, name, line=None, column=None):
        super().__init__(line, column)
        self.param_type = param_type # TypeNode or string
        self.name = name # Identifier
    def __repr__(self): return f"Parameter(type={self.param_type}, name={self.name})"

# --- Statements ---
class CompoundStatement(ASTNode):
    def __init__(self, statements, line=None, column=None):
        super().__init__(line, column)
        self.statements = statements
    def __repr__(self): return f"CompoundStatement({self.statements})"

class DeclarationStatement(ASTNode):
    # Note: In C, declarations can have multiple declarators (int x, *y;).
    # This basic version handles one name per declaration statement.
    def __init__(self, decl_type, name, initializer=None, line=None, column=None):
        super().__init__(line, column)
        self.decl_type = decl_type # TypeNode or string
        self.name = name # Identifier
        self.initializer = initializer # Optional expression node
    def __repr__(self): return f"DeclarationStatement(type={self.decl_type}, name={self.name}, init={self.initializer})"

class AssignmentStatement(ASTNode):
    def __init__(self, lvalue, expression, line=None, column=None):
        super().__init__(line, column)
        self.lvalue = lvalue # Typically Identifier, could be more complex (e.g., array access)
        self.expression = expression
    def __repr__(self): return f"AssignmentStatement(lvalue={self.lvalue}, expr={self.expression})"

class ExpressionStatement(ASTNode):
    """Represents an expression used as a statement (e.g., func(); x++;)."""
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

# --- NEW STATEMENT NODES ---
class ForStatement(ASTNode):
    def __init__(self, init, condition, update, body, line=None, column=None):
        super().__init__(line, column)
        self.init = init # Optional: DeclarationStatement or ExpressionStatement/Expression
        self.condition = condition # Optional: Expression
        self.update = update # Optional: Expression
        self.body = body # Statement
    def __repr__(self): return f"ForStatement(init={self.init}, cond={self.condition}, update={self.update}, body={self.body})"

class DoWhileStatement(ASTNode):
    def __init__(self, body, condition, line=None, column=None):
        super().__init__(line, column)
        self.body = body # Statement
        self.condition = condition # Expression
    def __repr__(self): return f"DoWhileStatement(body={self.body}, cond={self.condition})"

class BreakStatement(ASTNode):
     def __repr__(self): return "BreakStatement"

class ContinueStatement(ASTNode):
     def __repr__(self): return "ContinueStatement"

# CallStatement removed, use CallExpression within ExpressionStatement

class ReturnStatement(ASTNode):
    def __init__(self, value=None, line=None, column=None):
        super().__init__(line, column)
        self.value = value # Optional expression node
    def __repr__(self): return f"ReturnStatement(value={self.value})"

# --- Expressions ---
class Identifier(ASTNode):
    def __init__(self, name, line=None, column=None):
        super().__init__(line, column)
        self.name = name
    def __repr__(self): return f"Identifier(name='{self.name}')"

class IntegerLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        # Store original string value from lexer
        self.raw_value = value
        try:
            if value.lower().startswith('0x'):
                 self.value = int(value, 16)
            elif value.startswith('0') and len(value) > 1:
                 self.value = int(value, 8)
            else:
                 self.value = int(value)
        except ValueError:
            # Handle potential suffixes like L, U here if needed, or raise error
            self.value = int(value.rstrip('uUlL'), 10) # Basic suffix removal
            # Store suffix info?
    def __repr__(self): return f"IntegerLiteral(value={self.value})"

class FloatLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.raw_value = value
        try:
            # Basic suffix removal
             self.value = float(value.rstrip('fFlL'))
        except ValueError:
             self.value = float('nan') # Error case
    def __repr__(self): return f"FloatLiteral(value={self.value})"

class StringLiteral(ASTNode):
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        # Assume lexer already handled escapes
        self.value = value
    def __repr__(self): return f"StringLiteral(value={repr(self.value)})"

class CharLiteral(ASTNode): # NEW
    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
         # Assume lexer already handled escapes
        self.value = value
    def __repr__(self): return f"CharLiteral(value={repr(self.value)})"


class BinaryOp(ASTNode):
    def __init__(self, op, left, right, line=None, column=None):
        # Use operator token's location
        super().__init__(line, column)
        self.op = op # String representation of operator
        self.left = left
        self.right = right
    def __repr__(self): return f"BinaryOp(op='{self.op}', left={self.left}, right={self.right})"

# --- NEW EXPRESSION NODES ---
class UnaryOp(ASTNode):
    def __init__(self, op, operand, line=None, column=None):
         # Use operator token's location
        super().__init__(line, column)
        self.op = op # String representation of operator
        self.operand = operand
    def __repr__(self): return f"UnaryOp(op='{self.op}', operand={self.operand})"

class CallExpression(ASTNode):
    def __init__(self, function, args, line=None, column=None):
        # Use function identifier's location
        super().__init__(line, column)
        self.function = function # Identifier or potentially complex expression
        self.args = args # List of expression nodes
    def __repr__(self): return f"CallExpression(function={self.function}, args={self.args})"


class ArraySubscript(ASTNode):
    """ Represents array subscript expressions, e.g., arr[index]. """

    def __init__(self, array_expr, index_expr, line=None, column=None):
        # Location info typically corresponds to the '[' token
        super().__init__(line, column)
        # Expression that results in the array/pointer being accessed
        self.array_expression = array_expr
        # Expression inside the square brackets
        self.index_expression = index_expr

    def __repr__(self):
        return f"ArraySubscript(array={self.array_expression}, index={self.index_expression})"


class MemberAccess(ASTNode):
    """ Represents member access using '.' or '->'. """

    def __init__(self, object_or_pointer_expr, member_identifier, is_pointer, line=None, column=None):
        # Location info typically corresponds to the '.' or '->' token
        super().__init__(line, column)
        # Expression for the object or pointer before the operator
        self.object_or_pointer_expression = object_or_pointer_expr
        # Identifier node for the member name
        self.member_identifier = member_identifier
        # Boolean flag: True if access is via '->', False if via '.'
        self.is_pointer_access = is_pointer

    def __repr__(self):
        op = '->' if self.is_pointer_access else '.'
        return f"MemberAccess(object={self.object_or_pointer_expression}, member='{self.member_identifier.name}', op='{op}')"
