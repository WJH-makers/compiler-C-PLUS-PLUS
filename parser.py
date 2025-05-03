# coding=utf-8
import logging
import sys

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    # 导入 AST 节点定义和词法分析器
    # --- <<< Import CastExpression >>> ---
    from compiler_ast import (
        ASTNode, Program, FunctionDefinition, Parameter, CompoundStatement,
        DeclarationStatement, AssignmentStatement, ExpressionStatement,
        IfStatement, WhileStatement, ForStatement, DoWhileStatement,
        BreakStatement, ContinueStatement, ReturnStatement, Identifier,
        IntegerLiteral, FloatLiteral, StringLiteral, CharLiteral,
        BinaryOp, UnaryOp, CallExpression, ArraySubscript, MemberAccess,
        CastExpression  # Add CastExpression here
    )
    # --- <<< End Import CastExpression >>> ---
    from lexer import LexerError, Lexer  # 假设 Lexer 在这里导入
except ImportError as e:
    print(f"错误：无法导入所需的模块 (compiler_ast, lexer)。请确保它们存在且路径正确。\n{e}", file=sys.stderr)
    sys.exit(1)


class ParseError(Exception):
    """用于解析错误的自定义异常。"""
    def __init__(self, message, token):
        location = f"L{token.original_line}:C{token.column}" if token and hasattr(token,
                                                                                  'original_line') and token.original_line is not None else "UnknownLocation/EOF"
        token_repr = repr(token.value) if token and hasattr(token, 'value') else 'EOF'
        super().__init__(f"Parse Error at {location}: {message} (found {token_repr})")
        self.token = token


class Parser:
    """
    语法分析器类，负责将 Token 流转换为抽象语法树 (AST)。
    使用 Precedence Climbing (优先级爬升) 方法解析表达式。
    """
    def __init__(self, tokens):
        self.token_iter = iter(tokens)
        self.current_token = None
        self.peek_token = None
        self._advance()
        self._advance()

    def _advance(self):
        """将 Token 流向前推进一个位置。"""
        self.current_token = self.peek_token
        try:
            self.peek_token = next(self.token_iter)
            # DEBUG: print(f"Advanced: current={self.current_token}, peek={self.peek_token}")
        except StopIteration:
            self.peek_token = None

    def _error(self, message, token=None):
        """报告解析错误并抛出异常。"""
        error_token = token or self.current_token
        # DEBUG: print(f"Parse Error called with token: {error_token}")
        if error_token is None:
            # If current is None, try to use the last non-None token for location? Difficult.
            raise ParseError(message + " (at end of input)", None)
        # Ensure error uses a token that has location info if possible
        if not hasattr(error_token, 'original_line') or error_token.original_line is None:
            # If the error token lacks location, find the last one that had it?
            # For now, just raise with what we have.
            pass
        raise ParseError(message, error_token)


    def _check(self, token_type, value=None):
        """检查当前 Token 是否匹配指定的类型和（可选的）值。"""
        if not self.current_token or self.current_token.type == "EOF":
            return False
        type_match = self.current_token.type == token_type
        value_match = (value is None or self.current_token.value == value)
        # DEBUG: print(f"Check: Current={self.current_token}, TargetType={token_type}, TargetValue={value} -> TypeMatch={type_match}, ValueMatch={value_match}")
        return type_match and value_match

    def _check_peek(self, token_type, value=None):
        """检查下一个 Token 是否匹配指定的类型和（可选的）值。"""
        if not self.peek_token or self.peek_token.type == "EOF":
            return False
        return self.peek_token.type == token_type and \
            (value is None or self.peek_token.value == value)

    def _match(self, token_type, value=None):
        """如果当前 Token 匹配，则消耗它（前进）并返回 True，否则返回 False。"""
        if self._check(token_type, value):
            self._advance()
            return True
        return False

    def _consume(self, token_type, error_msg, value=None):
        """期望当前 Token 匹配，消耗它，并返回被消耗的 Token。如果不匹配则引发 ParseError。"""
        consumed_token = self.current_token  # Store before potential advance
        if not self._check(token_type, value):
            expected = f"'{value}' ({token_type})" if value else token_type
            found_type = self.current_token.type if self.current_token else "EOF"
            found_val = repr(self.current_token.value) if self.current_token else "EOF"
            self._error(f"{error_msg}. Expected {expected}, but found {found_val} ({found_type})",
                        token=self.current_token or consumed_token)
        self._advance()  # Consume the token only if it matched
        return consumed_token  # Return the token that was consumed

    # 运算符优先级和结合性定义
    PRECEDENCE = {
        '=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1,
        '?': 2,  # Ternary operator (requires special handling, not fully implemented here)
        '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7,
        '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9,
        '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12,
        # Unary operators have higher precedence handled in _parse_unary_expression
        # Postfix operators like (), [], ->, . have highest precedence handled in _parse_postfix_expression
    }
    RIGHT_ASSOC = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '?'}

    # --- 主要解析方法 ---

    def _check_type_start(self):
        """检查当前 Token 是否可以开始一个 C/C++ 类型说明符 (包括 string)。"""
        # Handles basic types, const/volatile qualifiers
        if not self.current_token or self.current_token.type == "EOF":
            return False
        if self.current_token.type == "KEYWORD" and \
                self.current_token.value in ["void", "char", "short", "int", "long", "float", "double",
                                             "signed", "unsigned", "const", "volatile", "_Bool",
                                             "string",  # Treat string as a type keyword here
                                             # Add struct, union, enum if needed
                                             ]:
            return True
        # TODO: Handle custom type names (typedefs, class/struct names) - requires symbol table access during parsing? Or mark as IDENTIFIER and check later?
        # For now, only keywords start types.
        return False

    def _parse_type(self):
        """解析基本类型说明符序列 (e.g., 'const unsigned long int')。"""
        type_token = self.current_token  # For error reporting start location
        if not self._check_type_start():
            self._error("Expected type specifier keyword (int, char, float, const, string, etc.)", token=type_token)
            return None  # Should not be reachable due to check before call usually

        type_parts = []
        # Keep consuming type-related keywords
        while self.current_token and self._check_type_start():
            type_parts.append(self.current_token.value)
            self._advance()

        if not type_parts:  # Should not happen if initial check passed
            self._error("Internal Parser Error: Failed to parse type specifier parts", token=type_token)
            return None

        # Basic reordering/validation could happen here (e.g., 'int long' -> 'long int')
        # For simplicity, just join them. Semantic analysis can validate later.
        parsed_type = " ".join(type_parts)
        logging.debug(f"Parsed type specifier: {parsed_type}")
        return parsed_type

    def parse_program(self):
        """解析整个程序（外部声明序列）。"""
        prog_start_token = self.current_token
        declarations = []
        parse_count = 0
        max_parse_attempts = 1000  # Safety break for potential infinite loops on errors

        while self.current_token and self.current_token.type != "EOF" and parse_count < max_parse_attempts:
            start_while_token = self.current_token  # Track token at loop start for stuck detection
            parse_count += 1
            logging.debug(f"Top-level parse loop iteration {parse_count}, current token: {self.current_token}")

            try:
                # Handle 'using namespace ... ;'
                if self._check('KEYWORD', 'using') and self._check_peek('KEYWORD', 'namespace'):
                    logging.debug("Found 'using namespace' directive.")
                    self._advance()  # using
                    self._advance()  # namespace
                    ns_token = self._consume('IDENTIFIER', "Expected identifier after 'using namespace'")
                    self._consume('PUNCTUATOR', "Expected ';' after 'using namespace' directive", value=';')
                    logging.info(f"Skipped 'using namespace {ns_token.value};'")
                    continue  # Skip to next top-level item

                # Handle external declarations (functions or global variables)
                elif self._check_type_start():
                    logging.debug("Found start of an external declaration (type detected).")
                    decl = self._parse_external_declaration()
                    if decl:  # Might return None on certain errors
                        declarations.append(decl)
                    # _parse_external_declaration consumes the trailing ';' or '}'
                    continue  # Move to next declaration

                # Handle stray semicolons at top level
                elif self._match('PUNCTUATOR', ';'):
                    logging.info("Ignoring empty top-level statement (stray semicolon).")
                    continue

                # If none of the above, it's unexpected
                else:
                    self._error(
                        f"Unexpected token at top level. Expected type specifier (int, void, etc.), 'using', or ';'")
                    # Error automatically raises exception

            except ParseError as e:
                print(e, file=sys.stderr)  # Print the specific parse error
                # --- Stuck detection and recovery ---
                # If we didn't advance past the token that caused the error, force advance once.
                if self.current_token == start_while_token and (
                        self.current_token and self.current_token.type != "EOF"):
                    logging.warning(f"Parser seems stuck on token {self.current_token}. Forcing advance.")
                    self._advance()
                    # Check if forced advance led to EOF
                    if not self.current_token or self.current_token.type == "EOF":
                        logging.info("EOF reached after forced advance.")
                        break  # Exit loop if EOF reached

                # Attempt to synchronize to the next likely declaration or statement start
                logging.info("Attempting recovery by synchronizing...")
                self._synchronize()
                # Check if synchronization reached EOF
                if not self.current_token or self.current_token.type == "EOF":
                    logging.info("EOF reached after synchronization.")
                    break  # Exit loop if EOF reached
                logging.info(f"Resynchronized at token: {self.current_token}. Continuing parse.")
                # Continue to the next iteration of the while loop

        if parse_count >= max_parse_attempts:
            logging.error(
                f"Parser stopped after reaching maximum iterations ({max_parse_attempts}), potentially stuck in a loop.")

        # Determine program node location based on the first token encountered
        line = prog_start_token.original_line if prog_start_token else 1
        col = prog_start_token.column if prog_start_token else 1
        return Program(declarations, line=line, column=col)

    def _synchronize(self):
        """尝试基本的错误恢复：跳过 token 直到找到一个合适同步点。"""
        skipped_tokens_log = []
        # Tokens that might indicate the start of a new top-level declaration or statement
        sync_keywords = {'if', 'while', 'for', 'do', 'return', 'switch', 'void', 'char', 'int', 'long', 'float',
                         'double', 'const', 'struct', 'class', 'using'}
        sync_punctuators = {';', '}'}  # Semicolon often ends declarations/statements, '}' ends blocks

        while self.current_token and self.current_token.type != "EOF":
            token = self.current_token
            # Log skipped token concisely
            token_str = f"<{token.type},{repr(token.value)}>"
            skipped_tokens_log.append(token_str)

            # Synchronization points:
            # 1. Semicolon (often ends declarations or statements)
            if token.type == 'PUNCTUATOR' and token.value == ';':
                self._advance()  # Consume the semicolon itself
                logging.info(f"Sync: Resynchronized after ';'. Skipped: {' '.join(skipped_tokens_log)}")
                return
            # 2. Closing brace (likely ends a function/block, might be start of next global item)
            if token.type == 'PUNCTUATOR' and token.value == '}':
                # Don't consume the '}', let the outer loop handle it if it's expected
                logging.info(f"Sync: Resynchronized before '}}'. Skipped: {' '.join(skipped_tokens_log)}")
                return
            # 3. Start of a type specifier (likely start of new declaration/function)
            if self._check_type_start():
                logging.info(
                    f"Sync: Resynchronized before type '{token.value}'. Skipped: {' '.join(skipped_tokens_log)}")
                return
            # 4. Common statement keywords
            if token.type == 'KEYWORD' and token.value in sync_keywords:
                logging.info(
                    f"Sync: Resynchronized before keyword '{token.value}'. Skipped: {' '.join(skipped_tokens_log)}")
                return

            # If not a sync point, advance
            self._advance()

        logging.info(f"Sync: Reached EOF while synchronizing. Skipped: {' '.join(skipped_tokens_log)}")


    def _parse_external_declaration(self):
        """解析全局变量或函数定义/原型。假定当前 token 是类型开始。"""
        start_token = self.current_token
        decl_type = self._parse_type()  # Consumes type keywords
        pointer_level = 0
        # Parse potential pointer stars '*'
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        full_type_str = decl_type + '*' * pointer_level

        # Expect an identifier (variable or function name)
        name_token = self._consume('IDENTIFIER', "Expected identifier after type specifier")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)

        start_line = start_token.original_line if start_token else None
        start_col = start_token.column if start_token else None

        # Check if it's a function (prototype or definition) or a variable
        if self._check('PUNCTUATOR', '('):  # Function-like syntax
            self._advance()  # Consume '('
            params = self._parse_parameter_list()
            self._consume('PUNCTUATOR', "Expected ')' after parameter list", value=')')

            if self._check('PUNCTUATOR', '{'):  # Function Definition
                logging.debug(f"Parsing function definition body for {name_identifier.name}")
                body = self._parse_compound_statement()
                return FunctionDefinition(full_type_str, name_identifier, params, body, line=start_line,
                                          column=start_col)
            elif self._match('PUNCTUATOR', ';'):  # Function Prototype
                logging.info(f"Parsed function prototype: {full_type_str} {name_identifier.name}(...)")
                # Represent prototype as a DeclarationStatement with a flag
                decl_node = DeclarationStatement(full_type_str, name_identifier, None, line=start_line,
                                                 column=start_col)
                setattr(decl_node, 'is_prototype', True)
                setattr(decl_node, 'prototype_params', params)  # Store parsed param info
                return decl_node
            else:
                self._error("Expected '{' to start function body or ';' for prototype after parameters")
                return None  # Should not be reached

        else:  # Global Variable Declaration
            initializer = None
            if self._match('OPERATOR', '='):
                initializer = self._parse_assignment_expression()  # Use appropriate precedence
            # Array declaration syntax T var[size]; would be handled here too if implemented
            # TODO: Add array declaration parsing logic if needed
            self._consume('PUNCTUATOR', "Expected ';' after global variable declaration", value=';')
            decl_node = DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line,
                                             column=start_col)
            setattr(decl_node, 'is_prototype', False)  # Mark as not a prototype
            return decl_node

    def _parse_statement(self):
        """解析单个语句。"""
        loc_token = self.current_token
        line = loc_token.original_line if loc_token else None
        col = loc_token.column if loc_token else None

        if self._check_type_start():
            # It's a local declaration
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
            self._consume('PUNCTUATOR', "Expected ';'", value=';')
            return BreakStatement(line=line, column=col)
        elif self._match('KEYWORD', 'continue'):
            self._consume('PUNCTUATOR', "Expected ';'", value=';')
            return ContinueStatement(line=line, column=col)
        elif self._match('PUNCTUATOR', ';'):
            # Empty statement
            logging.debug("Parsed empty statement.")
            return None  # Or a specific EmptyStatement node if preferred
        else:  # Should be an expression statement
            # Check for EOF or block end before trying to parse expression
            if not self.current_token or self.current_token.type == 'EOF' or self._check('PUNCTUATOR', '}'):
                if self._check('PUNCTUATOR', '}'):
                    self._error("Expected statement or expression, but found '}'")
                else:  # EOF
                    self._error("Unexpected end of input, expected statement or expression")
                return None  # Indicates error

            # Parse the expression
            expr = self._parse_expression()
            if expr is None:
                # Error should have been raised by _parse_expression
                # Add a fallback error message if needed
                self._error("Invalid expression found where statement expected", token=loc_token)
                return None

            # Expect a semicolon after the expression
            self._consume('PUNCTUATOR', "Expected ';' after expression statement", value=';')
            return ExpressionStatement(expr, line=line, column=col)

    def _parse_compound_statement(self):
        """解析语句块 {}。"""
        start_token = self.current_token
        start_line = start_token.original_line if start_token else None
        start_col = start_token.column if start_token else None
        self._consume('PUNCTUATOR', "Expected '{' to start compound statement", value='{')
        statements = []
        # Loop until the closing brace is the current token
        while not self._check('PUNCTUATOR', '}'):
            # Check for unexpected EOF inside the block
            if not self.current_token or self.current_token.type == 'EOF':
                self._error("Unexpected end of input, expected '}' to close block",
                            token=start_token)  # Error at starting brace
                break  # Avoid infinite loop

            stmt = self._parse_statement()
            if stmt is not None:  # Add statement unless it was an empty statement (;)
                statements.append(stmt)
            # If _parse_statement returned None due to an error, the exception should have been caught higher up
            # or the loop condition will eventually fail

        self._consume('PUNCTUATOR', "Expected '}' to end compound statement", value='}')
        return CompoundStatement(statements, line=start_line, column=start_col)

    def _parse_declaration_statement(self):
        """解析局部变量声明。假定当前 token 是类型开始。"""
        start_token = self.current_token
        decl_type = self._parse_type()
        pointer_level = 0
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        full_type_str = decl_type + '*' * pointer_level

        name_token = self._consume('IDENTIFIER', f"Expected variable name after type '{full_type_str}'")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)

        initializer = None
        if self._match('OPERATOR', '='):
            initializer = self._parse_assignment_expression()
            if initializer is None:  # Check if parsing initializer failed
                self._error("Invalid initializer expression", token=self.current_token)  # Error at token after '='
                # Decide whether to return partial node or None
        # TODO: Handle array declarations like int arr[10];

        self._consume('PUNCTUATOR', "Expected ';' after declaration statement", value=';')
        start_line = start_token.original_line if start_token else None
        start_col = start_token.column if start_token else None
        decl_node = DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line, column=start_col)
        setattr(decl_node, 'is_prototype', False)  # Local declarations are not prototypes
        return decl_node

    def _parse_if_statement(self, line, col):
        """解析 if-else 语句。假定 'if' 已被消耗。"""
        self._consume('PUNCTUATOR', "Expected '(' after 'if' keyword", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'if' statement"); return None  # Error recovery
        self._consume('PUNCTUATOR', "Expected ')' after 'if' condition", value=')')
        then_branch = self._parse_statement()
        if then_branch is None and not self._check('KEYWORD', 'else'):  # Handle empty then branch ; followed by else
            pass  # Allow if(...); else ... by checking if 'else' is next
            # If then_branch is None because of parse error, it should have raised

        else_branch = None
        if self._match('KEYWORD', 'else'):
            else_branch = self._parse_statement()

        return IfStatement(condition, then_branch, else_branch, line=line, column=col)

    def _parse_while_statement(self, line, col):
        """解析 while 循环。假定 'while' 已被消耗。"""
        self._consume('PUNCTUATOR', "Expected '(' after 'while' keyword", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'while' statement"); return None
        self._consume('PUNCTUATOR', "Expected ')' after 'while' condition", value=')')
        body = self._parse_statement()
        return WhileStatement(condition, body, line=line, column=col)

    def _parse_for_statement(self, line, col):
        """解析 for 循环。假定 'for' 已被消耗。"""
        self._consume('PUNCTUATOR', "Expected '(' after 'for' keyword", value='(')
        init = None
        # Parse initialization part
        if not self._check('PUNCTUATOR', ';'):
            if self._check_type_start():  # Declaration inside for init
                init = self._parse_declaration_statement()  # Consumes the trailing ';' itself! Need adjustment
                # HACK/Fix: _parse_declaration_statement consumes ';', but for needs it. Backtrack conceptually.
                # Let's modify parse_declaration to NOT consume the semicolon if called from here
                # Or, parse parts manually here. Let's try manual parsing:
                start_init_token = self.current_token
                init_type = self._parse_type()
                init_pointer_level = 0
                while self._match('OPERATOR', '*'): init_pointer_level += 1
                init_name_token = self._consume('IDENTIFIER', "Expected variable name in 'for' initializer")
                init_name_id = Identifier(init_name_token.value, line=init_name_token.original_line,
                                          column=init_name_token.column)
                init_full_type = init_type + '*' * init_pointer_level
                init_value = None
                if self._match('OPERATOR', '='):
                    init_value = self._parse_assignment_expression()
                init_line = start_init_token.original_line if start_init_token else None
                init_col = start_init_token.column if start_init_token else None
                # Create node but *don't* consume semicolon yet
                init = DeclarationStatement(init_full_type, init_name_id, init_value, line=init_line, column=init_col)
                setattr(init, 'is_prototype', False)
            else:  # Expression initializer
                init = self._parse_expression()  # Does not consume semicolon
        # Consume semicolon after initializer (or if initializer was empty)
        self._consume('PUNCTUATOR', "Expected ';' after 'for' initializer expression", value=';')

        # Parse condition part
        condition = None
        if not self._check('PUNCTUATOR', ';'):
            condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after 'for' condition expression", value=';')

        # Parse update part
        update = None
        if not self._check('PUNCTUATOR', ')'):
            update = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'for' clauses", value=')')

        # Parse loop body
        body = self._parse_statement()
        return ForStatement(init, condition, update, body, line=line, column=col)

    def _parse_do_while_statement(self, line, col):
        """解析 do-while 循环。假定 'do' 已被消耗。"""
        body = self._parse_statement()
        # Handle case where body is empty statement (just ';')
        if body is None and not self._check('KEYWORD', 'while'):
            self._error("Expected 'while' after 'do' body (body cannot be just ';')",
                        token=self.current_token)  # Or use token before semicolon
            return None

        if not self._match('KEYWORD', 'while'):
            self._error("Expected 'while' keyword after 'do' body")
            return None
        self._consume('PUNCTUATOR', "Expected '(' after 'while' in do-while", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'do-while' statement"); return None
        self._consume('PUNCTUATOR', "Expected ')' after 'do-while' condition", value=')')
        self._consume('PUNCTUATOR', "Expected ';' after 'do-while' statement", value=';')
        return DoWhileStatement(body, condition, line=line, column=col)

    def _parse_return_statement(self, line, col):
        """解析 return 语句。假定 'return' 已被消耗。"""
        value = None
        # If the next token is not ';', parse an expression
        if not self._check('PUNCTUATOR', ';'):
            value = self._parse_expression()
            # Check if expression parsing failed
            if value is None:
                self._error("Expected expression or ';' after 'return'")
                # Return a node even on error? Or None? Let's return node with None value.
                value = None  # Reset value if expression parsing failed badly
        # Expect and consume the semicolon
        self._consume('PUNCTUATOR', "Expected ';' after return statement", value=';')
        return ReturnStatement(value, line=line, column=col)

    def _parse_parameter_list(self):
        """解析函数参数列表。 handles (), (void), (type name, ...)。"""
        params = []
        # Handle empty parameter list '()'
        if self._check('PUNCTUATOR', ')'):
            return []  # Empty list, will be consumed by caller

        # Handle '(void)'
        if self._check('KEYWORD', 'void') and self._check_peek('PUNCTUATOR', ')'):
            self._advance()  # Consume 'void'
            # Caller will consume ')'
            return []  # Represent (void) as empty list

        # If not empty or void, parse first parameter
        params.append(self._parse_parameter())
        # Parse subsequent parameters separated by commas
        while self._match('PUNCTUATOR', ','):
            if self._check('PUNCTUATOR', ')'):  # Error: trailing comma before ')'
                self._error("Unexpected ')' after comma in parameter list.")
                break
            # Check for varargs '...'
            if self._check('OPERATOR', '...'):  # Assuming lexer yields '...' as OPERATOR
                # TODO: Handle varargs - add special marker to params?
                self._match('OPERATOR', '...')  # Consume '...'
                logging.warning("Varargs (...) handling not fully implemented in AST.")
                break  # Stop parsing params after ...
            params.append(self._parse_parameter())

        return params

    def _parse_parameter(self):
        """解析单个函数参数 (type [*] [name])。"""
        start_token = self.current_token
        param_type = self._parse_type()  # Consumes type keywords
        pointer_level = 0
        while self._match('OPERATOR', '*'):
            pointer_level += 1
        full_param_type = param_type + '*' * pointer_level

        name_identifier = None
        # Parameter name is optional (e.g., in prototypes or unnamed params)
        if self._check('IDENTIFIER'):
            name_token = self._consume('IDENTIFIER', "Internal check failed: expected IDENTIFIER")
            name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        # If no identifier, check if it's followed by ',' or ')', otherwise maybe error?
        elif not self._check('PUNCTUATOR', ')') and not self._check('PUNCTUATOR', ','):
            # Allow unnamed parameters like 'int, float' if followed by comma or )
            pass  # It's a valid unnamed parameter in this context
            # self._error("Expected parameter name, comma, or closing parenthesis", token=self.current_token)

        start_line = start_token.original_line if start_token else None
        start_col = start_token.column if start_token else None
        return Parameter(full_param_type, name_identifier, line=start_line, column=start_col)

    # --- 表达式解析 (Precedence Climbing) ---

    def _get_precedence(self, token):
        """获取二元运算符的优先级。"""
        if not token or token.type != 'OPERATOR':
            return -1
        return self.PRECEDENCE.get(token.value, -1)

    def _parse_expression(self, min_precedence=0):
        """使用优先级爬升解析二元和一元表达式。"""
        # --- <<< MODIFICATION: Parse unary first >>> ---
        # Parse the left-hand side, which could be a primary expression
        # or include prefix unary operators.
        lhs = self._parse_unary_expression()
        if lhs is None:
            # Error already reported by _parse_unary_expression or its callees
            return None
        # --- <<< END MODIFICATION >>> ---

        while True:
            # Look at the next operator
            op_token = self.current_token
            if not op_token or op_token.type != 'OPERATOR':
                # DEBUG: print(f"Expression loop break: Not an operator {op_token}")
                break  # Not an operator, end of this precedence level

            precedence = self._get_precedence(op_token)
            # DEBUG: print(f"Expression loop: Op={op_token.value}, Prec={precedence}, MinPrec={min_precedence}")

            # Check precedence relative to the current minimum
            if precedence < min_precedence:
                # DEBUG: print(f"Expression loop break: Precedence {precedence} < {min_precedence}")
                break  # Operator has lower precedence than current context allows

            # Determine associativity for next recursive call's precedence
            # Right-associative operators call recursively with same precedence
            # Left-associative operators call recursively with precedence + 1
            next_min_precedence = precedence + (1 if op_token.value not in self.RIGHT_ASSOC else 0)

            # Consume the operator
            self._advance()

            # Parse the right-hand side operand recursively
            rhs = self._parse_expression(next_min_precedence)
            if rhs is None:
                # Error should have been raised by recursive call
                # Provide context if needed
                self._error("Expected expression after binary operator", token=op_token)
                return None  # Stop parsing this expression branch

            # Combine LHS and RHS with the operator
            op_line = op_token.original_line if op_token else None
            op_col = op_token.column if op_token else None
            # Check for assignment operator LValue (Syntax check only)
            is_lvalue_syntax_ok = isinstance(lhs, (Identifier, ArraySubscript, MemberAccess)) or \
                                  (isinstance(lhs, UnaryOp) and lhs.op == '*')
            if op_token.value in self.RIGHT_ASSOC and op_token.value != '?' and not is_lvalue_syntax_ok:
                # Report syntax error, but still build node for potential semantic check
                self._error(f"Invalid left-hand side in assignment operator '{op_token.value}'",
                            token=op_token)  # Error points to operator

            # Build the BinaryOp node
            lhs = BinaryOp(op_token.value, lhs, rhs, line=op_line, column=op_col)
            # Continue loop to check for operators with potentially higher precedence
        return lhs

    def _parse_assignment_expression(self):
        """解析赋值表达式 (最低常规优先级)。"""
        # Precedence 1 includes assignment operators like =, += etc.
        return self._parse_expression(min_precedence=1)

    def _parse_unary_expression(self):
        """解析前缀一元运算符 ('+', '-', '!', '~', '++', '--', '*', '&') 和 sizeof。"""
        op_token = self.current_token
        # Check for prefix unary operators
        prefix_ops = ['-', '+', '!', '~', '++', '--', '*', '&']  # Note: ++/-- here are prefix
        if self._check('OPERATOR') and op_token.value in prefix_ops:
            op_str = op_token.value
            op_line = op_token.original_line if op_token else None
            op_col = op_token.column if op_token else None
            self._advance()  # Consume the operator

            # Map ++/-- to distinct prefix versions for AST
            ast_op = '++p' if op_str == '++' else '--p' if op_str == '--' else op_str

            # Recursively parse the operand (which could be another unary expression)
            operand = self._parse_unary_expression()
            if operand is None:
                self._error("Expected expression after unary operator", token=op_token)
                return None

            # --- Syntax LValue Checks for specific operators ---
            # Check if operand is syntactically an LValue
            is_operand_lvalue = isinstance(operand, (Identifier, ArraySubscript, MemberAccess)) or \
                                (isinstance(operand, UnaryOp) and operand.op == '*')

            if op_str in ['++', '--'] and not is_operand_lvalue:
                # Prefix ++/-- require an LValue operand
                self._error(f"Operand of prefix operator '{op_str}' must be an lvalue", token=op_token)
                # Continue building node for potential semantic analysis? Or return None? Let's build.

            if op_str == '&' and not is_operand_lvalue:
                # Address-of operator requires an LValue
                self._error(f"Cannot take the address of a non-lvalue using operator '&'", token=op_token)
                # Continue building node

            return UnaryOp(ast_op, operand, line=op_line, column=op_col)

        # Check for sizeof operator
        elif self._match('KEYWORD', 'sizeof'):
            start_sizeof_token = self.current_token  # Token *after* sizeof keyword
            sizeof_line = start_sizeof_token.original_line if start_sizeof_token else None
            sizeof_col = start_sizeof_token.column if start_sizeof_token else None
            target = None

            # Check for sizeof(type) vs sizeof expression
            if self._match('PUNCTUATOR', '('):
                if self._check_type_start():  # sizeof(type)
                    target_type = self._parse_type()  # Consumes type keywords
                    pointer_level = 0
                    while self._match('OPERATOR', '*'): pointer_level += 1
                    target = target_type + '*' * pointer_level  # Store type as string
                    self._consume('PUNCTUATOR', "Expected ')' after type in sizeof", value=')')
                else:  # sizeof(expression)
                    target = self._parse_expression()  # Parse the expression inside
                    self._consume('PUNCTUATOR', "Expected ')' after expression in sizeof", value=')')
            else:  # sizeof expression (without parentheses, applies to unary expression)
                target = self._parse_unary_expression()

            if target is None:
                self._error("Expected type or expression after 'sizeof'", token=start_sizeof_token)
                return None
            # Create UnaryOp node for sizeof
            return UnaryOp('sizeof', target, line=sizeof_line, column=sizeof_col)

        else:
            # If not a prefix operator or sizeof, parse postfix/primary expression
            return self._parse_postfix_expression()


    def _parse_postfix_expression(self):
        """解析后缀运算符：(), [], ++, --, ., ->"""
        # First, parse the base primary expression or potential prefix unary result
        expr = self._parse_primary_expression()
        if expr is None:
            return None  # Error already reported

        # Then, loop to handle any postfix operators attached to it
        while True:
            op_token = self.current_token
            if not op_token or op_token.type == "EOF":
                break  # No more tokens or no postfix operator

            op_line = op_token.original_line if op_token else None
            op_col = op_token.column if op_token else None

            if self._match('PUNCTUATOR', '('):  # Function Call: expr(...)
                args = self._parse_argument_list()
                self._consume('PUNCTUATOR', "Expected ')' after function call arguments", value=')')
                expr = CallExpression(expr, args, line=op_line, column=op_col)  # Update expr with CallExpression node
            elif self._match('OPERATOR', '++'):  # Postfix Increment: expr++
                # Syntax LValue check
                is_expr_lvalue = isinstance(expr, (Identifier, ArraySubscript, MemberAccess)) or (
                        isinstance(expr, UnaryOp) and expr.op == '*')
                if not is_expr_lvalue: self._error("Operand of postfix '++' must be an lvalue", token=op_token)
                expr = UnaryOp('p++', expr, line=op_line, column=op_col)  # Update expr
            elif self._match('OPERATOR', '--'):  # Postfix Decrement: expr--
                is_expr_lvalue = isinstance(expr, (Identifier, ArraySubscript, MemberAccess)) or (
                        isinstance(expr, UnaryOp) and expr.op == '*')
                if not is_expr_lvalue: self._error("Operand of postfix '--' must be an lvalue", token=op_token)
                expr = UnaryOp('p--', expr, line=op_line, column=op_col)  # Update expr
            elif self._match('PUNCTUATOR', '['):  # Array Subscript: expr[...]
                index_expr = self._parse_expression()
                if index_expr is None: self._error("Expected index expression inside '[]'",
                                                   token=op_token); break  # Stop if index fails
                self._consume('PUNCTUATOR', "Expected ']' after array index", value=']')
                expr = ArraySubscript(expr, index_expr, line=op_line, column=op_col)  # Update expr
            elif self._match('OPERATOR', '.'):  # Member Access: expr.member
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '.'")
                member_id = Identifier(member_token.value, line=member_token.original_line, column=member_token.column)
                expr = MemberAccess(expr, member_id, is_pointer=False, line=op_line, column=op_col)  # Update expr
            elif self._match('OPERATOR', '->'):  # Pointer Member Access: expr->member
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '->'")
                member_id = Identifier(member_token.value, line=member_token.original_line, column=member_token.column)
                expr = MemberAccess(expr, member_id, is_pointer=True, line=op_line, column=op_col)  # Update expr
            else:
                # Not a recognized postfix operator, break the loop
                break
        return expr  # Return the final expression node after all postfix ops applied

    # --- <<< MODIFIED _parse_primary_expression >>> ---
    def _parse_primary_expression(self):
        """解析字面量、标识符、括号表达式和 C 风格类型转换。"""
        token = self.current_token
        if not token or token.type == "EOF":
            self._error("Unexpected EOF expecting primary expression")
            return None
        line = token.original_line if token else None
        col = token.column if token else None
        node = None

        if self._check('INTEGER_LITERAL'):
            token = self._consume('INTEGER_LITERAL', "Internal error: integer literal expected")
            node = IntegerLiteral(token.value, line=line, column=col)
        elif self._check('FLOAT_LITERAL'):
            token = self._consume('FLOAT_LITERAL', "Internal error: float literal expected")
            node = FloatLiteral(token.value, line=line, column=col)
        elif self._check('STRING_LITERAL'):
            token = self._consume('STRING_LITERAL', "Internal error: string literal expected")
            node = StringLiteral(token.value, line=line, column=col)
        elif self._check('CHAR_LITERAL'):
            token = self._consume('CHAR_LITERAL', "Internal error: char literal expected")
            node = CharLiteral(token.value, line=line, column=col)
        elif self._check('IDENTIFIER'):
            token = self._consume('IDENTIFIER', "Internal error: identifier expected")
            node = Identifier(token.value, line=line, column=col)

        elif self._check('PUNCTUATOR', '('):  # Could be parentheses or C-style cast
            start_paren_token = self.current_token
            paren_line = start_paren_token.original_line if start_paren_token else None
            paren_col = start_paren_token.column if start_paren_token else None
            self._advance()  # Consume '('

            # Check if it looks like a C-style cast: ( type ) ...
            # Use peek_token to check for ')' immediately after the type
            if self._check_type_start():
                # Potential cast, need to look ahead for ')'
                # This requires a temporary parse or more complex lookahead.
                # Let's try a simpler approach: parse type, check for ')'.
                logging.debug(f"Possible cast detected starting with {self.current_token}")
                target_type = self._parse_type()  # Consumes type tokens
                pointer_level = 0
                while self._match('OPERATOR', '*'):
                    pointer_level += 1
                full_target_type = target_type + '*' * pointer_level

                # NOW, check if the *next* token is ')'
                if self._check('PUNCTUATOR', ')'):
                    self._advance()  # Consume ')'
                    # Successfully parsed (type), now parse the expression to cast
                    # Cast has high precedence, binds tightly to the following unary expression
                    expression_to_cast = self._parse_unary_expression()
                    if expression_to_cast is None:
                        # Use the token *after* the cast's ')' for error location if possible
                        self._error("Expected expression following cast operator '(type)'",
                                    token=self.current_token or start_paren_token)
                        return None
                    # Create the CastExpression node
                    logging.debug(f"Successfully parsed CastExpression: ({full_target_type}) ...")
                    node = CastExpression(full_target_type, expression_to_cast, line=paren_line, column=paren_col)
                else:
                    # It wasn't `(type)`, it was `(type ...` perhaps a function call returning a type? Or just `(type)` alone?
                    # Treat as a regular parenthesized expression that *happens* to start with a type keyword
                    # This might occur in complex expressions or error cases. We need to "put back" the type tokens.
                    # THIS IS HARD WITHOUT A PUSHBACK MECHANISM.
                    # For simplicity now, we'll assume if a type is followed by ')', it's a cast.
                    # If not followed by ')', assume it's a parenthesized expression that starts with something
                    # that happens to also be a type keyword (like an identifier that matches 'int').
                    # Let's raise an error for now if it's ambiguous like (int i = 0)
                    self._error(f"Ambiguous syntax or missing ')' after potential type cast '{full_target_type}'",
                                token=self.current_token or start_paren_token)
                    # We should backtrack or handle parenthesized expressions differently.
                    # ---->> REVISING: Let's parse as normal expression if ')' doesn't follow type <<----
                    # This means we need to "reset" the parser state to before _parse_type was called.
                    # This is complex. A simpler parser might parse (expr) first, then check if expr IS a type node.
                    # Given the current structure, let's stick to the explicit check for (type)<unary_expr>

                    # --- Fallback to regular parentheses if not (type) structure ---
                    # Note: The above block now handles the cast case. If we reach here after '('
                    # it means _check_type_start() was false.
                    node = self._parse_expression()  # Parse expression inside ()
                    if node is None:
                        self._error("Expected expression inside parentheses",
                                    token=self.current_token or start_paren_token)
                        return None
                    self._consume('PUNCTUATOR', "Expected ')' to close parenthesized expression", value=')')

            else:  # Not a type starts after '(', must be regular parenthesized expr
                node = self._parse_expression()
                if node is None:
                    self._error("Expected expression inside parentheses", token=self.current_token or start_paren_token)
                    return None
                self._consume('PUNCTUATOR', "Expected ')' to close parenthesized expression", value=')')
        else:
            # Token is not a literal, identifier, or '('
            self._error(
                f"Unexpected token '{token.value}' ({token.type}), expected primary expression (literal, identifier, or '(')",
                token=token)
            return None
        return node

    # --- <<< END OF MODIFIED _parse_primary_expression >>> ---

    def _parse_argument_list(self):
        """解析函数调用参数。 Handles empty list () and list with args."""
        args = []
        # If the next token is ')', it's an empty list
        if self._check('PUNCTUATOR', ')'):
            return args

        # Parse the first argument
        # Arguments are assignment expressions (lowest precedence before comma)
        expr = self._parse_assignment_expression()
        if expr is None:
            # Error should have been raised by _parse_assignment_expression
            self._error("Expected expression as function argument", token=self.current_token)
            return []  # Return empty list on error? Or propagate None? Let's return empty.
        args.append(expr)

        # Parse subsequent arguments separated by commas
        while self._match('PUNCTUATOR', ','):
            # Check for immediate ')' after comma (syntax error)
            if self._check('PUNCTUATOR', ')'):
                self._error("Unexpected ')' after comma in argument list.")
                break
            # Parse the next argument expression
            expr = self._parse_assignment_expression()
            if expr is None:
                self._error("Expected expression after comma in argument list")
                break  # Stop parsing args on error
            args.append(expr)

        return args


# --- AST 打印函数 (包含对 CastExpression 的处理) ---
def print_ast_tree(node, indent="", last=True, prefix=""):
    """递归打印 AST 树，包含 CastExpression 处理。"""
    if node is None:
        # Handle None nodes gracefully, perhaps indicating an optional child wasn't present
        # print(f"{indent}{'└── ' if last else '├── '}{prefix}None")
        return  # Or print nothing for None nodes

    connector = '└── ' if last else '├── '
    node_repr = ""
    children = []  # List of tuples: (prefix, child_node_or_list, is_list_flag)
    line = getattr(node, 'line', None);
    col = getattr(node, 'column', None)
    line_info = f"(L{line}:{col})" if line is not None and col is not None else f"(L{line})" if line is not None else "(NoLoc)"

    # --- Node Type Specific Representation ---
    if isinstance(node, Program):
        node_repr = f"Program {line_info}";
        children = [("declarations", node.declarations)]
    elif isinstance(node, FunctionDefinition):
        node_repr = f"FunctionDefinition: {node.name.name} (returns: {node.return_type}) {line_info}";
        children = [
            ("params", node.params), ("body", node.body)]
    elif isinstance(node, Parameter):
        name_str = node.name.name if node.name else "<unnamed>";
        node_repr = f"Parameter: {name_str} (type: {node.param_type}) {line_info}";
        children = [
            ("name", node.name)]  # Show name identifier node if exists
    elif isinstance(node, CompoundStatement):
        node_repr = f"CompoundStatement {line_info}";
        children = [("statements", node.statements)]
    elif isinstance(node, DeclarationStatement):
        proto_str = " (prototype)" if getattr(node, 'is_prototype',
                                              False) else "";
        node_repr = f"DeclarationStatement: {node.name.name} (type: {node.decl_type}){proto_str} {line_info}";
        children = [
            ("name", node.name), ("initializer", node.initializer)];  # prototype_params handled below if exists
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
        node_repr = f"IntegerLiteral: {node.value} (raw='{node.raw_value}') {line_info}"
    elif isinstance(node, FloatLiteral):
        node_repr = f"FloatLiteral: {node.value} (raw='{node.raw_value}') {line_info}"
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
        op_prefix = "type" if is_sizeof_type else "operand";
        node_repr = f"UnaryOp: '{op_display}' {line_info}";
        children = [
            (op_prefix, operand_val)]
    elif isinstance(node, CallExpression):
        node_repr = f"CallExpression {line_info}";
        children = [("function", node.function), ("arguments", node.args)]
    elif isinstance(node, ArraySubscript):
        node_repr = f"ArraySubscript {line_info}";
        children = [("array", node.array_expression),
                    ("index", node.index_expression)]
    elif isinstance(node, MemberAccess):
        op = '->' if node.is_pointer_access else '.';
        node_repr = f"MemberAccess: {op}{node.member_identifier.name} {line_info}";
        children = [
            ("object", node.object_or_pointer_expression), ("memberId", node.member_identifier)]
    # --- <<< Handle CastExpression >>> ---
    elif isinstance(node, CastExpression):
        node_repr = f"CastExpression: (to type='{node.target_type}') {line_info}";
        children = [
            ("expression", node.expression)]
    # --- <<< End Handle CastExpression >>> ---
    elif isinstance(node, ASTNode):  # Fallback for other potential AST nodes
        node_repr = f"{type(node).__name__} {line_info}";
        children = [(attr, v) for attr, v in vars(node).items() if isinstance(v, (ASTNode, list))]  # Basic inspection
    elif isinstance(node, str):  # Handle simple string children (like sizeof type operand)
        print(f"{indent}{connector}{prefix}String: '{node}'");
        return
    else:  # Handle other unexpected types
        print(f"{indent}{connector}{prefix}{repr(node)}");
        return

    # Add prototype parameters if they exist
    if hasattr(node, 'is_prototype') and node.is_prototype and hasattr(node, 'prototype_params'):
        children.append(("prototype_params", node.prototype_params))

    # Print the current node line
    print(f"{indent}{connector}{prefix}{node_repr}")

    # Prepare and print children
    new_indent = indent + ('    ' if last else '│   ')
    valid_children = []
    for child_prefix, child_node_or_list in children:
        if isinstance(child_node_or_list, list):
            # Filter out None items from lists (e.g., optional else branch)
            items = [item for item in child_node_or_list if item is not None]
            if items:  # Only add if list is not empty after filtering
                valid_children.append((child_prefix, items, True))
        elif child_node_or_list is not None:
            # Add single non-None children
            valid_children.append((child_prefix, child_node_or_list, False))

    child_count = len(valid_children)
    for i, (child_prefix, child_node_or_list, is_list) in enumerate(valid_children):
        is_last_child = (i == child_count - 1)
        current_prefix = f"{child_prefix}: " if child_prefix else ""  # Add prefix like 'body: '
        if is_list:  # Handle lists of children
            num_items = len(child_node_or_list)
            list_prefix = f"{child_prefix or 'items'}"  # Default prefix if none provided
            # Print list header
            print(
                f"{new_indent}{'└── ' if is_last_child else '├── '}{current_prefix}[List: {list_prefix}, {num_items} item(s)]")
            # Print items in the list
            list_indent = new_indent + ('    ' if is_last_child else '│   ')
            for j, item in enumerate(child_node_or_list):
                is_last_item = (j == num_items - 1)
                item_prefix = f"[{j}]: "  # Prefix for list items
                print_ast_tree(item, indent=list_indent, last=is_last_item, prefix=item_prefix)
        elif isinstance(child_node_or_list, ASTNode):  # Handle single ASTNode child
            print_ast_tree(child_node_or_list, indent=new_indent, last=is_last_child, prefix=current_prefix)
        else:  # Handle simple string child (like sizeof type)
            print(f"{new_indent}{'└── ' if is_last_child else '├── '}{current_prefix}{repr(child_node_or_list)}")


# --- Main execution block (for standalone testing) ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python parser.py <input_file.cpp>", file=sys.stderr)
        sys.exit(1)

    input_file_path = sys.argv[1]
    raw_code = None
    processed_code = None
    tokens = None
    ast = None
    line_map = {}
    had_errors = False
    is_effectively_empty = False

    # Stages 1, 2, 3 (Reading, Preprocessing, Lexing) remain the same as provided previously...
    # --- Stage 1: Reading File ---
    try:
        print(f"--- Stage 1: Reading File ---")
        print(f"Reading file: {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            raw_code = infile.read()
        print("File reading complete.")
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.", file=sys.stderr);
        had_errors = True
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr);
        had_errors = True

    # --- Stage 2: Preprocessing (Optional) ---
    if not had_errors:
        try:
            print(f"\n--- Stage 2: Preprocessing ---")
            from preprocess import BasicPreprocessor

            print("Running preprocessor...")
            preprocessor = BasicPreprocessor()
            processed_code, line_map = preprocessor.process(raw_code)
            print("Preprocessing complete.")
            # Optionally print processed code and map
        except ImportError:
            print("Warning: preprocess.py not found. Skipping preprocessing stage.");
            processed_code = raw_code;
            num_lines = raw_code.count('\n') + 1;
            line_map = {i: i for i in range(1, num_lines + 1)}
        except Exception as e:
            print(f"Error during preprocessing: {e}. Using raw code.", file=sys.stderr);
            processed_code = raw_code;
            num_lines = raw_code.count('\n') + 1;
            line_map = {i: i for i in range(1, num_lines + 1)}

    # --- Stage 3: Lexical Analysis ---
    token_list = []  # Ensure token_list is defined
    if not had_errors and processed_code is not None:
        try:
            print(f"\n--- Stage 3: Lexical Analysis ---")
            print("Starting lexical analysis...")
            lexer = Lexer(processed_code, line_map)  # Pass line map
            token_list = list(lexer.tokenize())  # Store as list
            print(f"Lexical analysis complete. Generated {len(token_list)} tokens.")
        except LexerError as e:
            print(e, file=sys.stderr);
            had_errors = True
        except Exception as e:
            print(f"Unexpected Lexer Error: {e}", file=sys.stderr);
            import traceback;

            traceback.print_exc();
            had_errors = True

    # --- Stage 4: Syntax Analysis (Parsing) ---
    if not had_errors:
        # Check if token list is effectively empty (only EOF or nothing)
        is_effectively_empty = not token_list or (len(token_list) == 1 and token_list[0].type == 'EOF')
        if is_effectively_empty:
            print("\nInput is effectively empty after lexing. Creating empty Program AST.")
            ast = Program([])  # Handle empty input gracefully
        else:
            try:
                print(f"\n--- Stage 4: Syntax Analysis (Parsing) ---")
                print("Starting syntax analysis...")
                # Pass the list (or an iterator of it) to the parser
                parser = Parser(iter(token_list))  # Pass iterator
                ast = parser.parse_program()  # Attempt parsing
                print("\n--- Abstract Syntax Tree (Generated by Parser) ---")
                # Use the defined print_ast_tree function
                print_ast_tree(ast) if ast else print("AST generation failed.")
                print("-------------------------------------------------")
                print("Syntax analysis complete.")
                # If parser returns None for non-empty input, it's an error
                if ast is None:
                    print("Error: Syntax analysis failed to produce an AST for non-empty input.", file=sys.stderr)
                    had_errors = True

            except ParseError as final_e:
                # Error message should have been printed by parser._error
                print(f"Syntax analysis failed.", file=sys.stderr)  # Simple final message
                had_errors = True
            except Exception as e:
                print(f"Unexpected error during syntax analysis: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                had_errors = True
    elif not had_errors and not token_list:  # Handle case where lexer produced no tokens
        print("\nLexer produced no tokens. Skipping parsing.")
        is_effectively_empty = True
        ast = Program([])  # Treat as empty program

    # --- Final Summary ---
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("Compilation failed due to errors in Lexing or Parsing.")
        sys.exit(1)
    else:
        print("Parsing completed successfully (Lexing and Parsing stages).")
        # You would typically proceed to semantic analysis here if parsing succeeded.
        print("(Semantic Analysis stage would run next)")
        sys.exit(0)
