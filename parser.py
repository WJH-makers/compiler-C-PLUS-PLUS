# coding=utf-8
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    from compiler_ast import *
    from lexer import LexerError, Lexer
except ImportError as e:
    print(f"错误：无法导入所需的模块 (compiler_ast, lexer)。请确保它们存在且路径正确。\n{e}", file=sys.stderr)
    sys.exit(1)


class ParseError(Exception):
    def __init__(self, message, token):
        location = f"L{token.original_line}:C{token.column}" if token and hasattr(token,
                                                                                  'original_line') and token.original_line is not None else "UnknownLocation/EOF"
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
        if not self.current_token or self.current_token.type == "EOF":
            return False
        return self.current_token.type == token_type and \
            (value is None or self.current_token.value == value)

    def _check_peek(self, token_type, value=None):
        """ Checks the next token type and optionally its value without consuming it. """
        if not self.peek_token or self.peek_token.type == "EOF":
            return False
        return self.peek_token.type == token_type and \
            (value is None or self.peek_token.value == value)

    def _match(self, token_type, value=None):
        """ If the current token matches, consumes it and returns True, otherwise False. """
        if self._check(token_type, value):
            self._advance()
            return True
        return False

    def _consume(self, token_type, error_msg, value=None):
        """ Expects the current token to match, consumes it, and returns it. Raises ParseError otherwise. """
        if not self._check(token_type, value):
            expected = f"'{value}' ({token_type})" if value else token_type
            found_type = self.current_token.type if self.current_token else "EOF"
            found_val = repr(self.current_token.value) if self.current_token else "EOF"
            self._error(f"{error_msg}. Expected {expected}, but found {found_val} ({found_type})")
        token = self.current_token
        self._advance()
        return token

    # Operator Precedence and Associativity definitions
    PRECEDENCE = {
        '=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1,
        '?': 2, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7,
        '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9,
        '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12,
    }
    RIGHT_ASSOC = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='}

    # --- Main Parsing Method ---
    def parse_program(self):
        """ Parses the entire program (a sequence of external declarations). """
        loc_token = self.current_token
        declarations = []
        parse_count = 0
        max_parse_attempts = 1000  # Safety break for potential infinite loops

        while self.current_token and self.current_token.type != "EOF" and parse_count < max_parse_attempts:
            start_while_token = self.current_token
            parse_count += 1
            try:
                # Handle 'using namespace' (Assumes KEYWORD type from lexer)
                if self._check('KEYWORD', 'using') and self._check_peek('KEYWORD', 'namespace'):
                    self._advance()
                    self._advance()
                    namespace_token = self._consume('IDENTIFIER', "Expected identifier after 'using namespace'")
                    self._consume('PUNCTUATOR', "Expected ';' after 'using namespace' directive", value=';')
                    logging.info(f"Skipping using namespace {namespace_token.value};")
                    continue
                # Handle Type Declarations (Functions/Global Vars)
                elif self._check_type_start():
                    declarations.append(self._parse_external_declaration())
                # Handle Empty Statements
                elif self._match('PUNCTUATOR', ';'):
                    logging.info("Ignoring empty declaration (extra semicolon) at top level.")
                    continue
                # Unexpected token at top level
                else:
                    self._error(f"Unexpected token at top level: Expected type specifier, 'using', or ';'")
            except ParseError as e:
                print(e, file=sys.stderr)  # Report error
                # Basic error recovery: force advance if stuck, then synchronize
                if self.current_token == start_while_token and self.current_token.type != "EOF":
                    logging.warning("Parser stuck on token after error. Forcing advance.")
                    self._advance()
                    if not self.current_token or self.current_token.type == "EOF": break
                self._synchronize()
                if not self.current_token or self.current_token.type == "EOF":
                    logging.info("Reached EOF after error recovery.")
                    break

        if parse_count >= max_parse_attempts:
            logging.error("Parser reached maximum parse attempts, likely stuck in a loop.")

        # Determine program node location from the first token
        line = loc_token.original_line if loc_token and hasattr(loc_token,
                                                                'original_line') and loc_token.original_line is not None else 1
        col = loc_token.column if loc_token else 1
        return Program(declarations, line=line, column=col)

    def _synchronize(self):
        logging.info("Attempting error recovery...")
        skipped_tokens = []
        recovery_keywords = {'if', 'while', 'for', 'do', 'return', 'switch', 'case', 'default', 'break', 'continue'}

        while self.current_token and self.current_token.type != "EOF":
            token = self.current_token
            skipped_tokens.append(str(token))

            if token.type == 'PUNCTUATOR' and token.value == ';':
                self._advance()
                logging.info(f"Sync: stopped after ';'. Skipped: {' '.join(skipped_tokens)}")
                return
            if self._check_type_start():
                logging.info(f"Sync: stopped before type '{token.value}'. Skipped: {' '.join(skipped_tokens)}")
                return
            if token.type == 'KEYWORD' and token.value in recovery_keywords:
                logging.info(f"Sync: stopped before keyword '{token.value}'. Skipped: {' '.join(skipped_tokens)}")
                return
            if token.type == 'PUNCTUATOR' and token.value == '}':
                logging.info(f"Sync: stopped before '}}'. Skipped: {' '.join(skipped_tokens)}")
                return
            self._advance()
        logging.info(f"Sync: reached EOF. Skipped: {' '.join(skipped_tokens)}")

    def _parse_external_declaration(self):
        start_token = self.current_token
        decl_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        name_token = self._consume('IDENTIFIER', "Expected identifier for function or global variable name")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        full_type_str = decl_type + '*' * pointer_level
        start_line = start_token.original_line if start_token and hasattr(start_token,
                                                                          'original_line') and start_token.original_line is not None else None
        start_col = start_token.column if start_token else None

        if self._check('PUNCTUATOR', '('):
            self._consume('PUNCTUATOR', "Expected '(' for function parameter list", value='(')
            params = self._parse_parameter_list()
            self._consume('PUNCTUATOR', "Expected ')' after function parameter list", value=')')
            # Definition or Declaration?
            if self._check('PUNCTUATOR', '{'):  # Definition
                body = self._parse_compound_statement()
                return FunctionDefinition(full_type_str, name_identifier, params, body, line=start_line,
                                          column=start_col)
            elif self._match('PUNCTUATOR', ';'):  # Declaration (Prototype)
                logging.warning(f"Function prototype '{name_token.value}' parsed (AST node is DeclarationStatement).")
                # Add a flag or use a different node type for prototypes if needed later
                return DeclarationStatement(full_type_str, name_identifier, None, is_prototype=True, line=start_line,
                                            column=start_col)
            else:
                self._error("Expected '{' for function body or ';' for function prototype after parameter list")
        else:
            initializer = None
            if self._match('OPERATOR', '='):
                initializer = self._parse_assignment_expression()
            self._consume('PUNCTUATOR', "Expected ';' after global variable declaration", value=';')
            return DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line, column=start_col)

    def _parse_statement(self):
        loc_token = self.current_token
        line = loc_token.original_line if loc_token and hasattr(loc_token,
                                                                'original_line') and loc_token.original_line is not None else None
        col = loc_token.column if loc_token else None

        if self._check_type_start():
            return self._parse_declaration_statement()
        elif self._check('PUNCTUATOR', '{'):
            return self._parse_compound_statement()
        elif self._match('KEYWORD', 'if'):
            return self._parse_if_statement(line, col)
        elif self._match('KEYWORD', 'while'):
            return self._parse_while_statement(line, col)
        elif self._match('KEYWORD', 'for'):
            return self._parse_for_statement(line, col)
        elif self._match('KEYWORD', 'do'):
            return self._parse_do_while_statement(line, col)
        elif self._match('KEYWORD', 'return'):
            return self._parse_return_statement(line, col)
        elif self._match('KEYWORD', 'break'):
            self._consume('PUNCTUATOR', "Expected ';' after 'break'", value=';')
            return BreakStatement(line=line, column=col)
        elif self._match('KEYWORD', 'continue'):
            self._consume('PUNCTUATOR', "Expected ';' after 'continue'", value=';')
            return ContinueStatement(line=line, column=col)
        elif self._match('PUNCTUATOR', ';'):
            return None
        else:
            expr = self._parse_expression()
            self._consume('PUNCTUATOR', "Expected ';' after expression statement", value=';')
            return ExpressionStatement(expr, line=line, column=col)

    def _parse_compound_statement(self):
        start_token = self.current_token
        start_line = start_token.original_line if start_token and hasattr(start_token,
                                                                          'original_line') and start_token.original_line is not None else None
        start_col = start_token.column if start_token else None
        self._consume('PUNCTUATOR', "Expected '{' to start a compound statement", value='{')
        statements = []
        while not self._check('PUNCTUATOR', '}'):
            # Check for premature EOF
            if not self.current_token or self.current_token.type == 'EOF':
                self._error("Expected '}' to close compound statement, but reached end of input",
                            token=start_token)  # Report error at opening brace
            stmt = self._parse_statement()
            if stmt is not None:  # Only add non-empty statements
                statements.append(stmt)
        self._consume('PUNCTUATOR', "Expected '}' to close compound statement", value='}')
        return CompoundStatement(statements, line=start_line, column=start_col)

    def _parse_declaration_statement(self):
        """ Parses a local variable declaration. """
        start_token = self.current_token
        decl_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        name_token = self._consume('IDENTIFIER', "Expected variable name in declaration")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        full_type_str = decl_type + '*' * pointer_level
        initializer = None
        # Check for optional initializer
        if self._match('OPERATOR', '='):
            initializer = self._parse_assignment_expression()  # Use assignment precedence for initializer
        self._consume('PUNCTUATOR', "Expected ';' after declaration statement", value=';')
        start_line = start_token.original_line if start_token and hasattr(start_token,
                                                                          'original_line') and start_token.original_line is not None else None
        start_col = start_token.column if start_token else None
        return DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line, column=start_col)

    def _parse_if_statement(self, line, col):
        """ Parses an if-else statement. """
        self._consume('PUNCTUATOR', "Expected '(' after 'if'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'if' condition", value=')')
        then_branch = self._parse_statement()
        else_branch = None
        if self._match('KEYWORD', 'else'):
            else_branch = self._parse_statement()
        return IfStatement(condition, then_branch, else_branch, line=line, column=col)

    def _parse_while_statement(self, line, col):
        """ Parses a while loop. """
        self._consume('PUNCTUATOR', "Expected '(' after 'while'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'while' condition", value=')')
        body = self._parse_statement()
        return WhileStatement(condition, body, line=line, column=col)

    def _parse_for_statement(self, line, col):
        """ Parses a for loop (init; condition; update). """
        self._consume('PUNCTUATOR', "Expected '(' after 'for'", value='(')
        init = None
        # --- Parse Initializer ---
        if not self._check('PUNCTUATOR', ';'):
            if self._check_type_start():  # Declaration initializer
                start_init_token = self.current_token
                init_type = self._parse_type()
                init_pointer_level = 0
                while self._match('OPERATOR', '*'): init_pointer_level += 1
                init_name_token = self._consume('IDENTIFIER', "Expected variable name in 'for' initializer declaration")
                init_name_id = Identifier(init_name_token.value, line=init_name_token.original_line,
                                          column=init_name_token.column)
                init_full_type = init_type + '*' * init_pointer_level
                init_value = None
                if self._match('OPERATOR', '='):
                    init_value = self._parse_assignment_expression()
                init_line = start_init_token.original_line if start_init_token and hasattr(start_init_token,
                                                                                           'original_line') and start_init_token.original_line is not None else None
                init_col = start_init_token.column if start_init_token else None
                init = DeclarationStatement(init_full_type, init_name_id, init_value, line=init_line, column=init_col)
            else:  # Expression initializer
                init = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after 'for' initializer", value=';')
        # --- Parse Condition ---
        condition = None
        if not self._check('PUNCTUATOR', ';'):
            condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after 'for' condition", value=';')
        # --- Parse Update ---
        update = None
        if not self._check('PUNCTUATOR', ')'):
            update = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'for' clauses", value=')')
        # --- Parse Body ---
        body = self._parse_statement()
        return ForStatement(init, condition, update, body, line=line, column=col)

    def _parse_do_while_statement(self, line, col):
        """ Parses a do-while loop. """
        body = self._parse_statement()
        if not self._match('KEYWORD', 'while'):
            self._error("Expected 'while' keyword after 'do' body")
        self._consume('PUNCTUATOR', "Expected '(' after 'do...while'", value='(')
        condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'do...while' condition", value=')')
        self._consume('PUNCTUATOR', "Expected ';' after 'do...while' statement", value=';')
        return DoWhileStatement(body, condition, line=line, column=col)

    def _parse_return_statement(self, line, col):
        """ Parses a return statement. """
        value = None
        # Check if there's a return value expression
        if not self._check('PUNCTUATOR', ';'):
            value = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after return statement", value=';')
        return ReturnStatement(value, line=line, column=col)

    # --- Parsing Types and Parameters ---
    def _check_type_start(self):
        """ Checks if current token starts a C/C++ type specifier. """
        # Add other type keywords like struct, enum, etc. if needed
        return self.current_token and \
            self.current_token.type == "KEYWORD" and \
            self.current_token.value in ["void", "char", "short", "int", "long", "float", "double",
                                         "signed", "unsigned", "const", "volatile", "_Bool", "_Complex", "_Imaginary"]

    def _parse_type(self):
        """ Parses consecutive type keywords into a string representation. """
        base_type_token = self.current_token
        if not self._check_type_start():
            self._error("Expected type specifier", token=base_type_token)
        type_keywords = []
        while self._check_type_start():
            type_keywords.append(self.current_token.value)
            self._advance()
        return " ".join(type_keywords)

    def _parse_parameter_list(self):
        """ Parses function parameter list: (type name, type name, ...). """
        params = []
        if not self._check('PUNCTUATOR', ')'):  # If parameter list is not empty
            params.append(self._parse_parameter())
            while self._match('PUNCTUATOR', ','):
                if self._check('PUNCTUATOR', ')'):
                    self._error("Unexpected ')' after comma in parameter list.")
                params.append(self._parse_parameter())
        return params

    def _parse_parameter(self):
        """ Parses a single function parameter: type [*] [name]. """
        start_token = self.current_token
        param_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        full_param_type = param_type + '*' * pointer_level
        name_identifier = None
        # Parameter name is optional
        if self._check('IDENTIFIER'):
            name_token = self._consume('IDENTIFIER', "Expected parameter name or end of parameter")
            name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        start_line = start_token.original_line if start_token and hasattr(start_token,
                                                                          'original_line') and start_token.original_line is not None else None
        start_col = start_token.column if start_token else None
        return Parameter(full_param_type, name_identifier, line=start_line, column=start_col)

    # --- Expression Parsing (Pratt Parser) ---
    def _get_precedence(self, token):
        """ Gets precedence of a binary operator token. """
        if not token or token.type != 'OPERATOR':
            return -1  # Not a binary operator
        return self.PRECEDENCE.get(token.value, -1)  # Return -1 if not in map

    def _parse_expression(self, min_precedence=0):
        """ Parses expression using precedence climbing algorithm. """
        lhs = self._parse_unary_expression()  # Parse the left-most part first

        while True:
            op_token = self.current_token
            # Stop if no token, not an operator, or precedence too low
            if not op_token:
                break
            precedence = self._get_precedence(op_token)
            if precedence < min_precedence:
                break

            # Determine precedence for the recursive call based on associativity
            next_precedence = precedence + (1 if op_token.value not in self.RIGHT_ASSOC else 0)

            self._advance()  # Consume the operator
            rhs = self._parse_expression(next_precedence)  # Recursively parse the right side

            # Basic LValue check for assignment operators
            if op_token.value in self.RIGHT_ASSOC and not isinstance(lhs, Identifier):  # Simplistic check
                self._error(f"Invalid left-hand side for assignment operator '{op_token.value}'", token=op_token)

            # Combine into a BinaryOp node
            op_line = op_token.original_line if op_token and hasattr(op_token,
                                                                     'original_line') and op_token.original_line is not None else None
            op_col = op_token.column if op_token else None
            lhs = BinaryOp(op_token.value, lhs, rhs, line=op_line, column=op_col)
        return lhs  # Return the fully parsed expression tree

    def _parse_assignment_expression(self):
        """ Parses assignment expressions (lowest precedence). """
        # Assignment has precedence 1
        return self._parse_expression(min_precedence=1)

    def _parse_unary_expression(self):
        """ Parses prefix unary operators (+, -, !, ~, ++, --, *, &) and sizeof. """
        op_token = self.current_token

        # Check for prefix operators
        if self._check('OPERATOR') and op_token.value in ['-', '+', '!', '~', '++', '--', '*', '&']:
            op_str = op_token.value
            op_line = op_token.original_line if hasattr(op_token,
                                                        'original_line') and op_token.original_line is not None else None
            op_col = op_token.column if op_token else None
            self._advance()  # Consume operator
            # Use distinct AST representation for prefix ++/-- if needed
            prefix_op_str = '++p' if op_str == '++' else '--p' if op_str == '--' else op_str
            # Recursively parse operand (allows chaining like --*p)
            operand = self._parse_unary_expression()
            # LValue checks for prefix ++/-- and address-of &
            if op_str in ['++', '--'] and not isinstance(operand, Identifier):  # Simplistic check
                self._error(f"Invalid operand for prefix operator '{op_str}': requires LValue", token=op_token)
            if op_str == '&' and not isinstance(operand, Identifier):  # Simplistic check
                self._error(f"Cannot take the address of a non-LValue with operator '&'", token=op_token)
            return UnaryOp(prefix_op_str, operand, line=op_line, column=op_col)

        # Check for sizeof operator
        elif self._match('KEYWORD', 'sizeof'):
            start_sizeof = self.current_token  # For location info
            sizeof_line = start_sizeof.original_line if hasattr(start_sizeof,
                                                                'original_line') and start_sizeof.original_line is not None else None
            sizeof_col = start_sizeof.column if start_sizeof else None
            target = None
            target_is_type = False
            # Check for sizeof(...) vs sizeof expression
            if self._match('PUNCTUATOR', '('):
                if self._check_type_start():  # sizeof(type)
                    target_type = self._parse_type()
                    pointer_level = 0
                    while self._match('OPERATOR', '*'): pointer_level += 1
                    target = target_type + '*' * pointer_level  # Represent type as string
                    target_is_type = True
                else:  # sizeof(expression)
                    target = self._parse_expression()
                self._consume('PUNCTUATOR', "Expected ')' after sizeof argument", value=')')
            else:  # sizeof expression (no parentheses)
                # Operand follows unary precedence rules
                target = self._parse_unary_expression()
            return UnaryOp('sizeof', target, line=sizeof_line, column=sizeof_col)

        # If not a prefix op or sizeof, parse postfix/primary expression
        else:
            return self._parse_postfix_expression()

    def _parse_postfix_expression(self):
        """ Parses postfix operators: (), [], ++, --, ., -> """
        # Start with the base primary expression
        expr = self._parse_primary_expression()

        # Handle chained postfix operations
        while True:
            loc_token = self.current_token
            # Fix for SyntaxError: Separate check and break
            if not loc_token:
                break  # Exit loop if no more tokens

            loc_line = loc_token.original_line if hasattr(loc_token,
                                                          'original_line') and loc_token.original_line is not None else None
            loc_col = loc_token.column if loc_token else None

            # Check for different postfix operators
            if self._match('PUNCTUATOR', '('):  # Function Call
                args = self._parse_argument_list()
                self._consume('PUNCTUATOR', "Expected ')' after function call arguments", value=')')
                expr = CallExpression(expr, args, line=loc_line, column=loc_col)  # Use '(' location
            elif self._match('OPERATOR', '++'):  # Postfix ++
                # LValue check needed
                if not isinstance(expr, Identifier):  # Simplistic check
                    self._error("Operand of postfix '++' must be an LValue", token=loc_token)
                expr = UnaryOp('p++', expr, line=loc_line, column=loc_col)  # Use '++' location
            elif self._match('OPERATOR', '--'):  # Postfix --
                # LValue check needed
                if not isinstance(expr, Identifier):  # Simplistic check
                    self._error("Operand of postfix '--' must be an LValue", token=loc_token)
                expr = UnaryOp('p--', expr, line=loc_line, column=loc_col)  # Use '--' location
            elif self._match('PUNCTUATOR', '['):  # Array Subscript Operator
                index_expr = self._parse_expression()  # Parse index expression
                self._consume('PUNCTUATOR', "Expected ']' after array subscript index", value=']')
                # Assuming you have defined ArraySubscript in compiler_ast.py
                # from compiler_ast import ArraySubscript
                expr = ArraySubscript(expr, index_expr, line=loc_line, column=loc_col)  # Use '[' location
            elif self._match('OPERATOR', '.'):  # Member Access Operator
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '.'")
                member_identifier = Identifier(member_token.value, line=member_token.original_line,
                                               column=member_token.column)
                # Assuming you have defined MemberAccess in compiler_ast.py
                # from compiler_ast import MemberAccess
                expr = MemberAccess(expr, member_identifier, is_pointer=False, line=loc_line,
                                    column=loc_col)  # Use '.' location
            elif self._match('OPERATOR', '->'):  # Pointer Member Access Operator
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '->'")
                member_identifier = Identifier(member_token.value, line=member_token.original_line,
                                               column=member_token.column)
                # Assuming you have defined MemberAccess in compiler_ast.py
                # from compiler_ast import MemberAccess
                expr = MemberAccess(expr, member_identifier, is_pointer=True, line=loc_line,
                                    column=loc_col)  # Use '->' location
            # elif self._match('OPERATOR', '.'): ...
            # elif self._match('OPERATOR', '->'): ...
            else:
                break  # No matching postfix operator found, exit loop
        return expr  # Return the final expression tree with all postfix ops applied

    def _parse_primary_expression(self):
        """ Parses literals, identifiers, and parenthesized expressions. """
        token = self.current_token
        if not token or token.type == "EOF":
            self._error("Unexpected end of input while parsing primary expression")

        line = token.original_line if hasattr(token, 'original_line') and token.original_line is not None else None
        col = token.column if hasattr(token, 'column') else None
        node = None

        # --- IMPORTANT: Verify these type strings match YOUR lexer's output ---
        # Example: If your lexer uses 'INT' for integers, change 'INTEGER_LITERAL' to 'INT'.
        if self._check('INTEGER_LITERAL'):
            token = self._consume('INTEGER_LITERAL', "Internal error: Expected INTEGER_LITERAL")
            node = IntegerLiteral(token.value, line=line, column=col)
        elif self._check('FLOAT_LITERAL'):
            token = self._consume('FLOAT_LITERAL', "Internal error: Expected FLOAT_LITERAL")
            node = FloatLiteral(token.value, line=line, column=col)
        elif self._check('STRING_LITERAL'):
            token = self._consume('STRING_LITERAL', "Internal error: Expected STRING_LITERAL")
            node = StringLiteral(token.value, line=line, column=col)
        elif self._check('CHAR_LITERAL'):
            token = self._consume('CHAR_LITERAL', "Internal error: Expected CHAR_LITERAL")
            node = CharLiteral(token.value, line=line, column=col)
        # --- End Verification Section ---

        elif self._check('IDENTIFIER'):
            token = self._consume('IDENTIFIER', "Internal error: Expected IDENTIFIER")
            node = Identifier(token.value, line=line, column=col)
        elif self._check('PUNCTUATOR', '('):
            self._consume('PUNCTUATOR', "Internal error: Expected '('", value='(')
            node = self._parse_expression(min_precedence=0)  # Parse nested expression
            self._consume('PUNCTUATOR', "Expected ')' to close parenthesized expression", value=')')
        else:
            # If none of the above match, it's an unexpected token in this context
            self._error(f"Unexpected token found while parsing primary expression: {token.value}", token=token)
        return node

    def _parse_argument_list(self):
        """ Parses function call arguments: (expr, expr, ...). """
        args = []
        if not self._check('PUNCTUATOR', ')'):  # Check if argument list is non-empty
            args.append(self._parse_assignment_expression())  # Arguments are expressions
            while self._match('PUNCTUATOR', ','):
                # Handle trailing comma error case like func(arg1,)
                if self._check('PUNCTUATOR', ')'):
                    self._error("Unexpected ')' after comma in argument list.")
                args.append(self._parse_assignment_expression())
        return args


# --- AST Printing Function (Assumed Correct) ---
def print_ast_tree(node, indent="", last=True, prefix=""):
    # [Same as previous version]
    if node is None: print(f"{indent}{'└── ' if last else '├── '}{prefix}None"); return
    connector = '└── ' if last else '├── '
    node_repr = ""
    children = []
    line = getattr(node, 'line', None)
    line_info = f"(L{line})" if line is not None else "(NoLoc)"
    if isinstance(node, Program):
        node_repr = f"Program {line_info}";
        children = [("declarations", node.declarations)]
    elif isinstance(node, FunctionDefinition):
        node_repr = f"FunctionDefinition: {node.name.name} (returns: {node.return_type}) {line_info}";
        children = [
            ("params", node.params), ("body", node.body)]
    elif isinstance(node, Parameter):
        name_str = node.name.name if node.name else "<unnamed>";
        node_repr = f"Parameter: {name_str} (type: {node.param_type}) {line_info}"
    elif isinstance(node, CompoundStatement):
        node_repr = f"CompoundStatement {line_info}";
        children = [("statements", node.statements)]
    elif isinstance(node, DeclarationStatement):
        proto_str = " (prototype)" if getattr(node, 'is_prototype',
                                              False) else "";
        node_repr = f"DeclarationStatement: {node.name.name} (type: {node.decl_type}){proto_str} {line_info}";
        children = [
            ("initializer", node.initializer)]
    elif isinstance(node, ExpressionStatement):
        node_repr = f"ExpressionStatement {line_info}";
        children = [("expression", node.expression)]
    elif isinstance(node, IfStatement):
        node_repr = f"IfStatement {line_info}";
        children = [("condition", node.condition), ("then", node.then_branch),
                    ("else", node.else_branch)]
    elif isinstance(node, WhileStatement):
        node_repr = f"WhileStatement {line_info}";
        children = [("condition", node.condition), ("body", node.body)]
    elif isinstance(node, ForStatement):
        node_repr = f"ForStatement {line_info}";
        children = [("init", node.init), ("condition", node.condition),
                    ("update", node.update), ("body", node.body)]
    elif isinstance(node, DoWhileStatement):
        node_repr = f"DoWhileStatement {line_info}";
        children = [("body", node.body), ("condition", node.condition)]
    elif isinstance(node, BreakStatement):
        node_repr = f"BreakStatement {line_info}"
    elif isinstance(node, ContinueStatement):
        node_repr = f"ContinueStatement {line_info}"
    elif isinstance(node, ReturnStatement):
        node_repr = f"ReturnStatement {line_info}";
        children = [("value", node.value)]
    elif isinstance(node, Identifier):
        node_repr = f"Identifier: {node.name} {line_info}"
    elif isinstance(node, IntegerLiteral):
        node_repr = f"IntegerLiteral: {node.value} (raw: {getattr(node, 'raw_value', '?')}) {line_info}"
    elif isinstance(node, FloatLiteral):
        node_repr = f"FloatLiteral: {node.value} (raw: {getattr(node, 'raw_value', '?')}) {line_info}"
    elif isinstance(node, StringLiteral):
        node_repr = f"StringLiteral: {repr(node.value)} {line_info}"
    elif isinstance(node, CharLiteral):
        node_repr = f"CharLiteral: {repr(node.value)} {line_info}"
    elif isinstance(node, BinaryOp):
        node_repr = f"BinaryOp: '{node.op}' {line_info}";
        children = [("left", node.left), ("right", node.right)]
    elif isinstance(node, UnaryOp):
        op_display = node.op;
        operand_val = node.operand;
        is_sizeof_type = op_display == 'sizeof' and isinstance(
            operand_val,
            str);
        operand_prefix = "type" if is_sizeof_type else "operand";
        node_repr = f"UnaryOp: '{op_display}' {line_info}";
        children = [
            (operand_prefix, operand_val)]
    elif isinstance(node, CallExpression):
        node_repr = f"CallExpression {line_info}";
        children = [("function", node.function), ("arguments", node.args)]
    elif isinstance(node, ASTNode):
        node_repr = f"{type(node).__name__} {line_info}";
        children = [(attr, v) for attr, v in vars(node).items() if
                    isinstance(v, (ASTNode, list))]
    elif isinstance(node, str):
        print(f"{indent}{connector}{prefix}String: '{node}'");
        return
    else:
        print(f"{indent}{connector}{prefix}{repr(node)}");
        return
    print(f"{indent}{connector}{prefix}{node_repr}")
    new_indent = indent + ('    ' if last else '│   ')
    valid_children = [];
    for child_prefix, child_node_or_list in children:
        if isinstance(child_node_or_list, list):
            items = [item for item in child_node_or_list if item is not None];
            valid_children.append(
                (child_prefix, items, True)) if items else None
        elif child_node_or_list is not None:
            valid_children.append((child_prefix, child_node_or_list, False))
        elif child_prefix:
            pass
    child_count = len(valid_children);
    for i, (child_prefix, child_node_or_list, is_list) in enumerate(valid_children):
        is_last_child = (i == child_count - 1);
        current_prefix = f"{child_prefix}: " if child_prefix else ""
        if is_list:
            num_items = len(child_node_or_list);
            list_prefix = f"{child_prefix or 'items'}";
            print(f"{new_indent}{'└── ' if is_last_child else '├── '}{current_prefix}[List: {list_prefix}]")
            list_indent = new_indent + ('    ' if is_last_child else '│   ')
            for j, item in enumerate(child_node_or_list): is_last_item = (
                    j == num_items - 1); item_prefix = f"[{j}]: "; print_ast_tree(item, indent=list_indent,
                                                                                  last=is_last_item,
                                                                                  prefix=item_prefix)
        else:
            print_ast_tree(child_node_or_list, indent=new_indent, last=is_last_child, prefix=current_prefix)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Standard setup and execution flow
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <input_file.cpp>", file=sys.stderr)
        sys.exit(1)

    input_file_path = sys.argv[1]
    raw_code = None
    processed_code = None
    tokens = None
    ast = None
    line_map = {}
    had_errors = False

    # Stage 1: Read File
    try:
        print(f"--- Stage 1: Reading File ---")
        print(f"正在读取文件: {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            raw_code = infile.read()
        print("文件读取完成.")
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file_path}' 未找到", file=sys.stderr)
        had_errors = True
    except Exception as e:
        print(f"读取文件时发生错误: {e}", file=sys.stderr)
        had_errors = True

    # Stage 2: Preprocessing (Optional)
    if not had_errors:
        try:
            print(f"\n--- Stage 2: Preprocessing ---")
            from preprocess import BasicPreprocessor  # Attempt to use preprocessor

            print("正在运行预处理器...")
            preprocessor = BasicPreprocessor()
            processed_code, line_map = preprocessor.process(raw_code)
            print("--- Preprocessed Code ---");
            print(processed_code.strip())
            print("--- Line Map (Processed -> Original) ---")
            if line_map:
                limit = 50  # Limit printing for large maps
                print("  " + "\n  ".join(f"{k} -> {v}" for k, v in list(sorted(line_map.items()))[:limit]))
                if len(line_map) > limit: print("  ...")
            else:
                print("  (No line mapping generated)")
            print("-------------------------");
            print("预处理完成.")
        except ImportError:
            # Fallback if preprocessor not found
            print("警告: 找不到 preprocess.py。跳过预处理阶段。")
            print("将使用原始代码进行词法分析。行号将基于原始文件。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}  # Create dummy map
        except Exception as e:
            # Fallback on preprocessor error
            print(f"预处理时发生错误: {e}. 继续使用原始代码。", file=sys.stderr)
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}

    # Stage 3: Lexical Analysis
    if not had_errors and processed_code is not None:
        try:
            print(f"\n--- Stage 3: Lexical Analysis ---")
            print("正在进行词法分析...")
            lexer = Lexer(processed_code, line_map)  # Pass line map to lexer
            tokens = lexer.tokenize()
            print(f"词法分析完成. 共生成 {len(tokens)} tokens.")
        except LexerError as e:
            print(e, file=sys.stderr);
            print("词法分析失败。", file=sys.stderr);
            had_errors = True
        except Exception as e:
            print(f"词法分析时发生意外错误: {e}", file=sys.stderr);
            import traceback;

            traceback.print_exc();
            had_errors = True

    # Stage 4: Parsing
    if not had_errors and tokens is not None:
        try:
            print(f"\n--- Stage 4: Syntax Analysis (Parsing) ---")
            print("开始语法分析...")
            parser = Parser(tokens)
            ast = parser.parse_program()  # Attempt to parse
            print("\n--- Abstract Syntax Tree (AST) ---")
            print_ast_tree(ast) if ast else print("AST generation failed or resulted in None.")
            print("----------------------------------")
            print("语法分析过程结束.")
        except ParseError as final_e:
            print(f"语法分析失败 (见上方 Parse Error)。", file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"语法分析时发生意外错误: {e}", file=sys.stderr)
            import traceback;

            traceback.print_exc();
            had_errors = True
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("编译过程中检测到错误。无法生成完整 AST。")
        sys.exit(1)  # Exit with error code
    else:
        is_empty_input = len(tokens) <= 1  # Only EOF token means empty input
        has_declarations = ast and hasattr(ast, 'declarations') and ast.declarations

        if ast and (has_declarations or is_empty_input):
            print("编译成功完成 (预处理(可选), 词法分析, 语法分析).")
            sys.exit(0)  # Exit successfully
        elif ast is None and is_empty_input:
            print("编译过程结束 (输入为空或只包含注释/指令)。")
            sys.exit(0)
        else:
            print("编译过程结束，但未成功生成预期的 AST 结构。")
            sys.exit(1)
