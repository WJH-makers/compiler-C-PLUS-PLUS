# parser.py (Modified to use token.original_line consistently)
import sys
import logging

# Import AST nodes and Lexer (ensure paths are correct)
try:
    # Import specific AST node classes defined in compiler_ast.py
    from compiler_ast import (
        ASTNode, Program, FunctionDefinition, Parameter, CompoundStatement,
        DeclarationStatement, ExpressionStatement, IfStatement, WhileStatement,
        ForStatement, DoWhileStatement, BreakStatement, ContinueStatement,
        ReturnStatement, Identifier, IntegerLiteral, FloatLiteral, StringLiteral,
        CharLiteral, BinaryOp, UnaryOp, CallExpression
        # Add ArraySubscript, MemberAccess here if you define them later
    )
    # Import Lexer and LexerError from your modified lexer.py
    from lexer import LexerError, Lexer
except ImportError as e:
    print("错误：无法导入所需的模块 (compiler_ast, lexer)。请确保它们存在且路径正确。")
    print(e)
    sys.exit(1)

# Custom Parser Error using token.original_line
class ParseError(Exception):
    def __init__(self, message, token):
        location = f"L{token.original_line}:C{token.column}" if token and hasattr(token, 'original_line') else "UnknownLocation/EOF"
        token_repr = repr(token.value) if token and hasattr(token, 'value') else 'EOF'
        super().__init__(f"Parse Error at {location}: {message} (found {token_repr})")
        self.token = token

class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current_token = None
        self.peek_token = None
        self._advance()
        self._advance()

    def _advance(self):
        self.current_token = self.peek_token
        try:
            self.peek_token = next(self.tokens)
        except StopIteration:
            self.peek_token = None

    def _error(self, message, token=None):
        raise ParseError(message, token or self.current_token)

    def _check(self, token_type, value=None):
        if not self.current_token or self.current_token.type == "EOF": return False
        return self.current_token.type == token_type and \
            (value is None or self.current_token.value == value)

    def _check_peek(self, token_type, value=None):
        if not self.peek_token or self.peek_token.type == "EOF": return False
        return self.peek_token.type == token_type and \
            (value is None or self.peek_token.value == value)

    def _match(self, token_type, value=None):
        if self._check(token_type, value):
            self._advance()
            return True
        return False

    def _consume(self, token_type, error_msg, value=None):
        if not self._check(token_type, value):
            expected = f"'{value}' ({token_type})" if value else token_type
            self._error(f"{error_msg}. Expected {expected}")
        token = self.current_token
        self._advance()
        return token

    # Operator Precedence (Unchanged)
    PRECEDENCE = {
        '=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1,
        '?': 2, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7,
        '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9,
        '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12,
    }
    RIGHT_ASSOC = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='}

    # --- Main Parsing Method ---
    def parse_program(self):
        loc_token = self.current_token
        declarations = []
        while self.current_token and self.current_token.type != "EOF":
            start_while_token = self.current_token
            try:
                 if self._check('IDENTIFIER', 'using') and self._check_peek('IDENTIFIER', 'namespace'):
                      self._advance(); self._advance()
                      namespace_token = self._consume('IDENTIFIER', "Expected identifier after 'using namespace'")
                      self._consume('PUNCTUATOR', "Expected ';' after 'using namespace' directive", value=';')
                      logging.info(f"Skipping using namespace {namespace_token.value};")
                      continue
                 elif self._check_type_start():
                      declarations.append(self._parse_external_declaration())
                 elif self._match('PUNCTUATOR', ';'):
                      logging.info("Ignoring empty declaration (extra semicolon) at top level.")
                      continue
                 else:
                      self._error(f"Unexpected token at top level: Expected type specifier, 'using', or ';'")
            except ParseError as e:
                print(e, file=sys.stderr)
                if self.current_token == start_while_token and self.current_token.type != "EOF":
                     logging.warning("Parser stuck on token after error. Forcing advance.")
                     self._advance()
                     if not self.current_token or self.current_token.type == "EOF": break
                self._synchronize()
                if not self.current_token or self.current_token.type == "EOF":
                     logging.info("Reached EOF after error recovery."); break

        # Use original_line from the first token for the Program node's location
        line = loc_token.original_line if loc_token and hasattr(loc_token, 'original_line') else 1
        col = loc_token.column if loc_token else 1
        return Program(declarations, line=line, column=col)

    # Error Recovery (Unchanged)
    def _synchronize(self):
        logging.info("Attempting error recovery...")
        skipped_tokens = []
        while self.current_token and self.current_token.type != "EOF":
            skipped_tokens.append(str(self.current_token))
            if self.current_token.type == 'PUNCTUATOR' and self.current_token.value == ';':
                self._advance(); logging.info(f"Sync: stopped after ';'. Skipped: {' '.join(skipped_tokens)}"); return
            if self._check_type_start():
                logging.info(f"Sync: stopped before type '{self.current_token.value}'. Skipped: {' '.join(skipped_tokens)}"); return
            if self._check('PUNCTUATOR', '}'):
                logging.info(f"Sync: stopped before '}}'. Skipped: {' '.join(skipped_tokens)}"); return
            if self._check('KEYWORD') and self.current_token.value in ['if', 'while', 'for', 'do', 'return', 'switch', 'case', 'default']:
                 logging.info(f"Sync: stopped before keyword '{self.current_token.value}'. Skipped: {' '.join(skipped_tokens)}"); return
            self._advance()
        logging.info(f"Sync: reached EOF. Skipped: {' '.join(skipped_tokens)}")

    # External Declaration Parsing
    def _parse_external_declaration(self):
        start_token = self.current_token
        decl_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        name_token = self._consume('IDENTIFIER', "Expected identifier")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column) # Uses .original_line
        full_type_str = decl_type + '*' * pointer_level

        start_line = start_token.original_line if start_token and hasattr(start_token, 'original_line') else None # Uses .original_line
        start_col = start_token.column if start_token else None

        if self._check('PUNCTUATOR', '('): # Function
            self._consume('PUNCTUATOR', "Expected '('", value='(')
            params = self._parse_parameter_list()
            self._consume('PUNCTUATOR', "Expected ')'", value=')')
            if self._check('PUNCTUATOR', '{'): # Definition
                body = self._parse_compound_statement()
                return FunctionDefinition(full_type_str, name_identifier, params, body, line=start_line, column=start_col) # Uses start_line
            elif self._match('PUNCTUATOR', ';'): # Declaration (Prototype)
                logging.warning(f"Function prototype '{name_token.value}' parsed.")
                return DeclarationStatement(full_type_str, name_identifier, None, line=start_line, column=start_col) # Uses start_line
            else: self._error("Expected '{' for function body or ';' for prototype")
        else: # Global Variable
            initializer = None
            if self._match('OPERATOR', '='): initializer = self._parse_assignment_expression()
            self._consume('PUNCTUATOR', "Expected ';' after global var", value=';')
            return DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line, column=start_col) # Uses start_line

    # Statement Parsing
    def _parse_statement(self):
        loc_token = self.current_token
        line = loc_token.original_line if loc_token and hasattr(loc_token, 'original_line') else None # Uses .original_line
        col = loc_token.column if loc_token else None

        if self._check_type_start(): return self._parse_declaration_statement()
        elif self._check('PUNCTUATOR', '{'): return self._parse_compound_statement()
        elif self._match('KEYWORD', 'if'): return self._parse_if_statement(line, col) # Pass line/col
        elif self._match('KEYWORD', 'while'): return self._parse_while_statement(line, col) # Pass line/col
        elif self._match('KEYWORD', 'for'): return self._parse_for_statement(line, col) # Pass line/col
        elif self._match('KEYWORD', 'do'): return self._parse_do_while_statement(line, col) # Pass line/col
        elif self._match('KEYWORD', 'return'): return self._parse_return_statement(line, col) # Pass line/col
        elif self._match('KEYWORD', 'break'):
            self._consume('PUNCTUATOR', "Expected ';' after 'break'", value=';')
            return BreakStatement(line=line, column=col) # Uses derived line/col
        elif self._match('KEYWORD', 'continue'):
            self._consume('PUNCTUATOR', "Expected ';' after 'continue'", value=';')
            return ContinueStatement(line=line, column=col) # Uses derived line/col
        elif self._match('PUNCTUATOR', ';'): return None
        else:
            expr = self._parse_expression()
            self._consume('PUNCTUATOR', "Expected ';' after expr statement", value=';')
            return ExpressionStatement(expr, line=line, column=col) # Uses derived line/col

    # Compound Statement Parsing
    def _parse_compound_statement(self, is_function_body=False):
        start_token = self.current_token
        self._consume('PUNCTUATOR', "Expected '{'", value='{')
        statements = []
        while not self._check('PUNCTUATOR', '}'):
             if not self.current_token or self.current_token.type == 'EOF': self._error("Expected '}'", token=start_token)
             stmt = self._parse_statement()
             if stmt: statements.append(stmt)
        self._consume('PUNCTUATOR', "Expected '}'", value='}')
        line = start_token.original_line if start_token and hasattr(start_token, 'original_line') else None # Uses .original_line
        col = start_token.column if start_token else None
        return CompoundStatement(statements, line=line, column=col)

    # Declaration Parsing
    def _parse_declaration_statement(self):
        start_token = self.current_token
        decl_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        name_token = self._consume('IDENTIFIER', "Expected variable name")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column) # Uses .original_line
        full_type_str = decl_type + '*' * pointer_level
        initializer = None
        if self._match('OPERATOR', '='): initializer = self._parse_assignment_expression()
        self._consume('PUNCTUATOR', "Expected ';'", value=';')
        start_line = start_token.original_line if start_token and hasattr(start_token, 'original_line') else None # Uses .original_line
        start_col = start_token.column if start_token else None
        return DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line, column=start_col)

    # Other Statement Parsers (use line/col passed in)
    def _parse_if_statement(self, line, col):
        self._consume('PUNCTUATOR', "Expected '(' after 'if'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after condition", value=')')
        then_branch = self._parse_statement()
        else_branch = None
        if self._match('KEYWORD', 'else'): else_branch = self._parse_statement()
        return IfStatement(condition, then_branch, else_branch, line=line, column=col)

    def _parse_while_statement(self, line, col):
        self._consume('PUNCTUATOR', "Expected '(' after 'while'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after condition", value=')')
        body = self._parse_statement()
        return WhileStatement(condition, body, line=line, column=col)

    def _parse_for_statement(self, line, col):
         self._consume('PUNCTUATOR', "Expected '(' after 'for'", value='(')
         init = None
         if not self._check('PUNCTUATOR', ';'):
             if self._check_type_start():
                 start_init_token = self.current_token
                 init_type = self._parse_type()
                 init_pointer_level = 0
                 while self._match('OPERATOR', '*'): init_pointer_level += 1
                 init_name_token = self._consume('IDENTIFIER', "Expected var name in 'for' init")
                 init_name_id = Identifier(init_name_token.value, line=init_name_token.original_line, column=init_name_token.column) # Uses .original_line
                 init_full_type = init_type + '*' * init_pointer_level
                 init_value = None
                 if self._match('OPERATOR', '='): init_value = self._parse_assignment_expression()
                 init_line = start_init_token.original_line if start_init_token and hasattr(start_init_token, 'original_line') else None # Uses .original_line
                 init_col = start_init_token.column if start_init_token else None
                 init = DeclarationStatement(init_full_type, init_name_id, init_value, line=init_line, column=init_col)
             else: init = self._parse_expression()
         self._consume('PUNCTUATOR', "Expected ';' after 'for' init", value=';')
         condition = None
         if not self._check('PUNCTUATOR', ';'): condition = self._parse_expression()
         self._consume('PUNCTUATOR', "Expected ';' after 'for' condition", value=';')
         update = None
         if not self._check('PUNCTUATOR', ')'): update = self._parse_expression()
         self._consume('PUNCTUATOR', "Expected ')' after 'for' clauses", value=')')
         body = self._parse_statement()
         return ForStatement(init, condition, update, body, line=line, column=col)

    def _parse_do_while_statement(self, line, col):
        body = self._parse_statement()
        if not self._match('KEYWORD', 'while'): self._error("Expected 'while' after 'do' body")
        self._consume('PUNCTUATOR', "Expected '(' after 'do...while'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after condition", value=')')
        self._consume('PUNCTUATOR', "Expected ';' after 'do...while'", value=';')
        return DoWhileStatement(body, condition, line=line, column=col)

    def _parse_return_statement(self, line, col):
        value = None
        if not self._check('PUNCTUATOR', ';'): value = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after return", value=';')
        return ReturnStatement(value, line=line, column=col)

    # Type and Parameter Parsing
    def _check_type_start(self): # Logic unchanged
        return self.current_token and \
            self.current_token.type == "KEYWORD" and \
            self.current_token.value in ["void","char","short","int","long","float","double",
                                         "signed","unsigned","const","volatile",
                                         "_Bool","_Complex","_Imaginary"]

    def _parse_type(self): # Logic unchanged
        base_type_token = self.current_token
        if not self._check_type_start(): self._error("Expected type specifier", token=base_type_token)
        type_keywords = []
        while self._check_type_start(): type_keywords.append(self.current_token.value); self._advance()
        base_type = " ".join(type_keywords)
        return base_type

    def _parse_parameter_list(self): # Logic unchanged
        params = []
        if not self._check('PUNCTUATOR', ')'):
            params.append(self._parse_parameter())
            while self._match('PUNCTUATOR', ','):
                if self._check('PUNCTUATOR', ')'): self._error("Unexpected ')' after comma")
                params.append(self._parse_parameter())
        return params

    def _parse_parameter(self):
        start_token = self.current_token
        param_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        full_param_type = param_type + '*' * pointer_level
        name_identifier = None
        if self._check('IDENTIFIER'):
            name_token = self._consume('IDENTIFIER', "Expected parameter name")
            name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column) # Uses .original_line
        start_line = start_token.original_line if start_token and hasattr(start_token, 'original_line') else None # Uses .original_line
        start_col = start_token.column if start_token else None
        return Parameter(full_param_type, name_identifier, line=start_line, column=start_col)

    # Expression Parsing
    def _get_precedence(self, token): # Logic unchanged
        if not token or token.type != 'OPERATOR': return -1
        return self.PRECEDENCE.get(token.value, -1)

    def _parse_expression(self, min_precedence=0):
        lhs = self._parse_unary_expression()
        while True:
            op_token = self.current_token
            precedence = self._get_precedence(op_token)
            if precedence < min_precedence: break
            next_precedence = precedence + (1 if op_token.value not in self.RIGHT_ASSOC else 0)
            self._advance()
            rhs = self._parse_expression(next_precedence)
            if op_token.value in self.RIGHT_ASSOC: # Basic lvalue check
                if not isinstance(lhs, Identifier): self._error(f"Invalid LHS for assignment '{op_token.value}'", token=op_token)
            op_line = op_token.original_line if op_token and hasattr(op_token, 'original_line') else None # Uses .original_line
            op_col = op_token.column if op_token else None
            lhs = BinaryOp(op_token.value, lhs, rhs, line=op_line, column=op_col)
        return lhs

    def _parse_assignment_expression(self): # Logic unchanged
        return self._parse_expression(min_precedence=1)

    def _parse_unary_expression(self):
        op_token = self.current_token
        op_line = op_token.original_line if op_token and hasattr(op_token, 'original_line') else None # Uses .original_line
        op_col = op_token.column if op_token else None

        if self._check('OPERATOR') and op_token.value in ['-', '+', '!', '~', '++', '--', '*', '&']:
            op_str = op_token.value
            self._advance()
            prefix_op_str = '++p' if op_str == '++' else '--p' if op_str == '--' else op_str
            operand = self._parse_unary_expression()
            if op_str in ['++', '--', '&']: # Basic lvalue check
                if not isinstance(operand, Identifier): self._error(f"Invalid operand for prefix '{op_str}'", token=op_token)
            return UnaryOp(prefix_op_str, operand, line=op_line, column=op_col) # Use op_line/op_col
        elif self._match('KEYWORD', 'sizeof'):
            start_sizeof = op_token
            sizeof_line = start_sizeof.original_line if start_sizeof and hasattr(start_sizeof, 'original_line') else None # Uses .original_line
            sizeof_col = start_sizeof.column if start_sizeof else None
            target = None; target_is_type = False
            if self._match('PUNCTUATOR', '('):
                 if self._check_type_start(): # sizeof(type)
                      target_type = self._parse_type()
                      pointer_level = 0
                      while self._match('OPERATOR', '*'): pointer_level += 1
                      target = target_type + '*' * pointer_level # Pass type string
                      target_is_type = True
                 else: target = self._parse_expression() # sizeof(expression)
                 self._consume('PUNCTUATOR', "Expected ')' after sizeof", value=')')
            else: target = self._parse_unary_expression() # sizeof expression
            return UnaryOp('sizeof', target, line=sizeof_line, column=sizeof_col) # Use sizeof_line/col
        else:
            return self._parse_postfix_expression()

    def _parse_postfix_expression(self):
        expr = self._parse_primary_expression()
        while True:
            loc_token = self.current_token
            if not loc_token: break

            loc_line = loc_token.original_line if loc_token and hasattr(loc_token, 'original_line') else None # Uses .original_line
            loc_col = loc_token.column if loc_token else None

            if self._match('PUNCTUATOR', '('): # Call
                args = self._parse_argument_list()
                self._consume('PUNCTUATOR', "Expected ')' after args", value=')')
                expr = CallExpression(expr, args, line=loc_line, column=loc_col) # Use loc_line/loc_col
            # ArraySubscript and MemberAccess parsing commented out as nodes aren't defined in user's AST
            # elif self._match('PUNCTUATOR', '['): ...
            elif self._match('OPERATOR', '++'): # Postfix ++
                 if not isinstance(expr, Identifier): self._error("Operand of postfix '++' must be lvalue", token=loc_token)
                 expr = UnaryOp('p++', expr, line=loc_line, column=loc_col) # Use loc_line/loc_col
            elif self._match('OPERATOR', '--'): # Postfix --
                 if not isinstance(expr, Identifier): self._error("Operand of postfix '--' must be lvalue", token=loc_token)
                 expr = UnaryOp('p--', expr, line=loc_line, column=loc_col) # Use loc_line/loc_col
            # elif self._match('OPERATOR', '.'): ...
            # elif self._match('OPERATOR', '->'): ...
            else: break
        return expr

    def _parse_primary_expression(self):
        token = self.current_token
        if not token: self._error("Unexpected EOF in expression")
        line = token.original_line if hasattr(token, 'original_line') else None # Uses .original_line
        col = token.column if hasattr(token, 'column') else None

        if self._check('INTEGER'):
            token = self._consume('INTEGER', "Internal error"); return IntegerLiteral(token.value, line=line, column=col)
        elif self._check('FLOAT'):
            token = self._consume('FLOAT', "Internal error"); return FloatLiteral(token.value, line=line, column=col)
        elif self._check('STRING'):
            token = self._consume('STRING', "Internal error"); return StringLiteral(token.value, line=line, column=col)
        elif self._check('CHAR'):
            token = self._consume('CHAR', "Internal error"); return CharLiteral(token.value, line=line, column=col)
        elif self._check('IDENTIFIER'):
            token = self._consume('IDENTIFIER', "Internal error"); return Identifier(token.value, line=line, column=col)
        elif self._check('PUNCTUATOR', '('):
            paren_token = self._consume('PUNCTUATOR', "Internal error", value='(')
            expr = self._parse_expression()
            self._consume('PUNCTUATOR', "Expected ')' after expression", value=')')
            return expr # Inner expression carries its own location info
        else:
            self._error(f"Unexpected token in primary expression: {token.value}", token=token)

    def _parse_argument_list(self): # Logic unchanged
        args = []
        if not self._check('PUNCTUATOR', ')'):
            args.append(self._parse_assignment_expression())
            while self._match('PUNCTUATOR', ','):
                if self._check('PUNCTUATOR', ')'): self._error("Unexpected ')' after comma")
                args.append(self._parse_assignment_expression())
        return args

# AST Printing Function (Relies on node.line being populated correctly)
def print_ast_tree(node, indent="", last=True, prefix=""):
    # (Function Body remains the same as in the previous correct version)
    if node is None: print(f"{indent}{'└── ' if last else '├── '}{prefix}None"); return
    connector = '└── ' if last else '├── '
    node_repr = ""; children = []
    line_info = f"(L{node.line})" if hasattr(node, 'line') and node.line is not None else "(NoLoc)"
    if isinstance(node, Program): node_repr = f"Program {line_info}"; children = [("declarations", node.declarations)]
    elif isinstance(node, FunctionDefinition): node_repr = f"FunctionDefinition: {node.name.name} (returns: {node.return_type}) {line_info}"; children = [("params", node.params), ("body", node.body)]
    elif isinstance(node, Parameter): name_str = node.name.name if node.name else "<unnamed>"; node_repr = f"Parameter: {name_str} (type: {node.param_type}) {line_info}"
    elif isinstance(node, CompoundStatement): node_repr = f"CompoundStatement {line_info}"; children = [("statements", node.statements)]
    elif isinstance(node, DeclarationStatement): node_repr = f"DeclarationStatement: {node.name.name} (type: {node.decl_type}) {line_info}"; children = [("initializer", node.initializer)]
    elif isinstance(node, ExpressionStatement): node_repr = f"ExpressionStatement {line_info}"; children = [("expression", node.expression)]
    elif isinstance(node, IfStatement): node_repr = f"IfStatement {line_info}"; children = [("condition", node.condition), ("then", node.then_branch), ("else", node.else_branch)]
    elif isinstance(node, WhileStatement): node_repr = f"WhileStatement {line_info}"; children = [("condition", node.condition), ("body", node.body)]
    elif isinstance(node, ForStatement): node_repr = f"ForStatement {line_info}"; children = [("init", node.init), ("condition", node.condition), ("update", node.update), ("body", node.body)]
    elif isinstance(node, DoWhileStatement): node_repr = f"DoWhileStatement {line_info}"; children = [("body", node.body), ("condition", node.condition)]
    elif isinstance(node, BreakStatement): node_repr = f"BreakStatement {line_info}"
    elif isinstance(node, ContinueStatement): node_repr = f"ContinueStatement {line_info}"
    elif isinstance(node, ReturnStatement): node_repr = f"ReturnStatement {line_info}"; children = [("value", node.value)]
    elif isinstance(node, Identifier): node_repr = f"Identifier: {node.name} {line_info}"
    elif isinstance(node, IntegerLiteral): node_repr = f"IntegerLiteral: {node.value} (raw: {node.raw_value}) {line_info}"
    elif isinstance(node, FloatLiteral): node_repr = f"FloatLiteral: {node.value} (raw: {node.raw_value}) {line_info}"
    elif isinstance(node, StringLiteral): node_repr = f"StringLiteral: {repr(node.value)} {line_info}"
    elif isinstance(node, CharLiteral): node_repr = f"CharLiteral: {repr(node.value)} {line_info}"
    elif isinstance(node, BinaryOp): node_repr = f"BinaryOp: '{node.op}' {line_info}"; children = [("left", node.left), ("right", node.right)]
    elif isinstance(node, UnaryOp):
        op_display = node.op; operand_prefix = "type" if op_display == 'sizeof' and isinstance(node.operand, str) else "operand"
        node_repr = f"UnaryOp: '{op_display}' {line_info}"; children = [(operand_prefix, node.operand)]
    elif isinstance(node, CallExpression): node_repr = f"CallExpression {line_info}"; children = [("function", node.function), ("arguments", node.args)]
    elif isinstance(node, ASTNode): node_repr = f"{type(node).__name__} {line_info}"; children = [(attr, v) for attr, v in vars(node).items() if isinstance(v, (ASTNode, list))]
    elif isinstance(node, str): print(f"{indent}{connector}{prefix}String: '{node}'"); return
    else: print(f"{indent}{connector}{prefix}{repr(node)}"); return
    print(f"{indent}{connector}{prefix}{node_repr}")
    new_indent = indent + ('    ' if last else '│   ')
    valid_children = []
    for child_prefix, child_node_or_list in children:
         if isinstance(child_node_or_list, list): items = [item for item in child_node_or_list if item is not None]; valid_children.append((child_prefix, items, True)) if items else None
         elif child_node_or_list is not None: valid_children.append((child_prefix, child_node_or_list, False))
         elif child_prefix: valid_children.append((child_prefix, None, False))
    child_count = len(valid_children)
    for i, (child_prefix, child_node_or_list, is_list) in enumerate(valid_children):
        is_last_child = (i == child_count - 1); current_prefix = f"{child_prefix}: " if child_prefix else ""
        if is_list:
             num_items = len(child_node_or_list)
             for j, item in enumerate(child_node_or_list): is_last_item = (j == num_items - 1); item_prefix = f"{child_prefix}[{j}]: " if child_prefix else f"[{j}]: "; print_ast_tree(item, indent=new_indent, last=is_last_item, prefix=item_prefix)
        else: print_ast_tree(child_node_or_list, indent=new_indent, last=is_last_child, prefix=current_prefix)

# Main Execution Block (Handles preprocessor -> lexer -> parser flow)
if __name__ == "__main__":
    if len(sys.argv) != 2: print(f"用法: python {sys.argv[0]} <input_file.cpp>", file=sys.stderr); sys.exit(1)
    input_file_path = sys.argv[1]; raw_code = None; processed_code = None; tokens = None; ast = None; line_map = {}; had_errors = False
    # Stage 1: Read File
    try:
        # try 块内的语句需要缩进
        print(f"正在读取文件: {input_file_path}")
        # with 语句也需要相对于 try 缩进
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            # with 块内的语句需要相对于 with 缩进
            raw_code = infile.read()
    # except 需要与 try 对齐
    except FileNotFoundError:
        # except 块内的语句需要缩进
        print(f"错误: 文件 '{input_file_path}' 未找到", file=sys.stderr)
        had_errors = True  # 可以放在同一块内，新起一行
    # 另一个 except 需要与上一个 except (或 try) 对齐
    except Exception as e:
        # except 块内的语句需要缩进
        print(f"读取文件时发生错误: {e}", file=sys.stderr)
        had_errors = True
    # Stage 2: Preprocessing
    if not had_errors:
        try: from preprocess import BasicPreprocessor; print("正在运行预处理器--------"); preprocessor = BasicPreprocessor(); processed_code, line_map = preprocessor.process(raw_code); print("--- Preprocessed Code ---"); print(processed_code); print("--- Line Map (Processed -> Original) ---"); print("  " + "\n  ".join(f"{k} -> {v}" for k, v in sorted(line_map.items()))) if line_map else print("  (No mapping generated)"); print("-------------------------"); print("预处理完成.")
        except ImportError: print("警告: 找不到 preprocess.py..."); processed_code = raw_code; num_lines = raw_code.count('\n') + 1; line_map = {i: i for i in range(1, num_lines + 1)}; print("使用原始代码和虚拟行号映射。")
        except Exception as e: print(f"预处理时发生错误: {e}...", file=sys.stderr); processed_code = raw_code; num_lines = raw_code.count('\n') + 1; line_map = {i: i for i in range(1, num_lines + 1)}; print("使用原始代码和虚拟行号映射。"); # Consider setting had_errors=True
    # Stage 3: Lexical Analysis
    if not had_errors and processed_code is not None:
        try: print("正在进行词法分析----------"); lexer = Lexer(processed_code, line_map); tokens = lexer.tokenize(); print(f"词法分析完成. 共生成 {len(tokens)} tokens.")
        except LexerError as e: print(e, file=sys.stderr); print("词法分析失败。", file=sys.stderr); had_errors = True
        except Exception as e: print(f"词法分析时发生意外错误: {e}", file=sys.stderr); import traceback; traceback.print_exc(); had_errors = True
    # Stage 4: Parsing
    if not had_errors and tokens is not None:
        try: print("开始语法分析----------"); parser = Parser(tokens); ast = parser.parse_program(); print("--- Abstract Syntax Tree (AST) ---"); print_ast_tree(ast) if ast else print("AST generation failed or resulted in None."); print("----------------------------------"); print("语法分析过程结束.");
        except ParseError as final_e: print(f"语法分析失败: {final_e}", file=sys.stderr); had_errors = True
        except Exception as e: print(f"语法分析时发生意外错误: {e}", file=sys.stderr); import traceback; traceback.print_exc(); had_errors = True
    # Final Status
    if had_errors: print("\n编译过程中检测到错误。"); sys.exit(1)
    else:
         if ast and (ast.declarations or len(tokens) <= 1): print("\n编译成功完成 (预处理, 词法分析, 语法分析)."); sys.exit(0)
         elif ast is None and len(tokens) <=1: print("\n编译过程结束 (输入为空或只包含注释/指令)。"); sys.exit(0)
         else: print("\n编译过程结束，但未成功生成完整 AST。"); sys.exit(1)