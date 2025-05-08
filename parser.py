# coding=utf-8
import logging
import sys

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s (%(filename)s:%(lineno)d): %(message)s')

try:
    from compiler_ast import *
    from lexer import LexerError, Lexer
except ImportError as e:
    print(f"错误：无法导入所需的模块 (compiler_ast, lexer)。请确保它们存在且路径正确。\n{e}", file=sys.stderr)
    sys.exit(1)


cpp_declaration_starters = [
    "alignas", "auto", "bool", "char", "class", "const", "consteval",
    "constexpr", "constinit", "decltype", "double", "enum", "extern",
    "float", "int", "long", "register", "restrict", "short", "signed",
    "static", "static_assert", "struct", "thread_local", "typedef",
    "union", "unsigned", "using", "void", "volatile", "wchar_t", "string"
]


class ParseError(Exception):
    def __init__(self, message, token):
        location = f"L{token.original_line}:C{token.column}" if token and hasattr(token,
                                                                                  'original_line') and token.original_line is not None else "UnknownLocation/EOF"
        token_repr = repr(token.value) if token and hasattr(token, 'value') else (
            'EOF' if token is None else repr(token))
        token_type_info = f" ({token.type})" if token and hasattr(token, 'type') else ""
        super().__init__(f"Parse Error at {location}: {message} (found {token_repr}{token_type_info})")
        self.token = token


class Parser:
    def __init__(self, tokens):
        self.token_iter = iter(tokens)
        self.current_token = None
        self.peek_token = None
        self._last_consumed_token_for_error = None
        self._advance()
        self._advance()
        self.std_namespace_is_active = False

    # --- Token 处理辅助方法 ---
    def _advance(self):
        self._last_consumed_token_for_error = self.current_token
        self.current_token = self.peek_token
        try:
            self.peek_token = next(self.token_iter)
        except StopIteration:
            self.peek_token = None
        logging.debug(
            f"[_advance] Consumed: {self._last_consumed_token_for_error}, Current: {self.current_token}, Peek: {self.peek_token}")

    def _error(self, message, token=None):
        error_token = token or self.current_token or self._last_consumed_token_for_error
        raise ParseError(message, error_token)

    def _check(self, token_type, value=None):
        if not self.current_token or self.current_token.type == "EOF": return False
        return self.current_token.type == token_type and (value is None or self.current_token.value == value)

    def _check_peek(self, token_type, value=None):
        if not self.peek_token or self.peek_token.type == "EOF": return False
        return self.peek_token.type == token_type and (value is None or self.peek_token.value == value)

    def _match(self, token_type, value=None):
        if self._check(token_type, value):
            logging.debug(f"[_match] Matched and consuming: {self.current_token}")
            self._advance()
            return True
        return False

    def _consume(self, token_type, error_msg, value=None):
        consumed_token = self.current_token
        if not self._check(token_type, value):
            expected = f"'{value}' ({token_type})" if value else token_type
            found_type = self.current_token.type if self.current_token else "EOF"
            found_val = repr(self.current_token.value) if self.current_token else "EOF"
            self._error(f"{error_msg}. Expected {expected}, but found {found_val} ({found_type})",
                        token=self.current_token or consumed_token)
        logging.debug(f"[_consume] Consuming: {consumed_token}")
        self._advance()
        return consumed_token

    # --- 运算符优先级与类型检查 (同上) ---
    PRECEDENCE = {'=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1,
                  '?': 2, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7, '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9,
                  '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12, }
    RIGHT_ASSOC = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '?'}

    def _check_type_start(self):
        logging.debug(f"[_check_type_start] Current token: {self.current_token}")
        if not self.current_token or self.current_token.type == "EOF": return False
        if self.current_token.type == "KEYWORD" and self.current_token.value in cpp_declaration_starters: return True
        if self.current_token.type == "IDENTIFIER" and self.current_token.value == "std" and self._check_peek(
                "OPERATOR", "::"): return True
        logging.debug(f"[_check_type_start] Result: False (Token '{self.current_token}' not a type start)")
        return False

    def _parse_type(self):
        type_token_start = self.current_token
        logging.debug(f"[_parse_type] Starting with token: {type_token_start}")
        type_parts = []
        if self.current_token and self.current_token.type == "IDENTIFIER" and self.current_token.value == "std" and self.peek_token and self.peek_token.type == "OPERATOR" and self.peek_token.value == "::":
            std_token_val = self.current_token.value
            self._advance()
            colon_token_val = self.current_token.value
            self._advance()
            if self.current_token and self.current_token.type == "KEYWORD" and self.current_token.value in cpp_declaration_starters:
                type_parts.extend(
                    [std_token_val, colon_token_val, self.current_token.value])
                self._advance()
                logging.debug(
                    f"[_parse_type] Parsed qualified type part: {''.join(type_parts)}")
            else:
                self._error(f"Expected known type keyword after 'std::', found {self.current_token}",
                            token=self.current_token)
                return None
        while self.current_token and self._check_type_start():
            if self.current_token.value == "std" and type_parts and type_parts[0] == "std": break
            logging.debug(f"[_parse_type] Adding part: {self.current_token.value}")
            type_parts.append(self.current_token.value)
            self._advance()
        if not type_parts: self._error("Expected type specifier keyword", token=type_token_start); return None
        parsed_type = " ".join(type_parts)
        if "::" in parsed_type: parsed_type = parsed_type.replace(" :: ", "::")
        logging.debug(f"[_parse_type] Parsed final type: '{parsed_type}'")
        return parsed_type

    # --- 顶层解析与错误恢复  ---
    def parse_program(self):
        prog_start_token = self.current_token
        declarations = []
        parse_count = 0
        max_parse_attempts = 1000
        self.std_namespace_is_active = False
        while self.current_token and self.current_token.type != "EOF" and parse_count < max_parse_attempts:
            start_while_token = self.current_token
            parse_count += 1
            logging.debug(
                f"[parse_program] Iteration {parse_count}, current: {self.current_token}, std_active: {self.std_namespace_is_active}")
            try:
                if self._check('KEYWORD', 'using') and self._check_peek('KEYWORD', 'namespace'):
                    logging.debug("[parse_program] Found 'using namespace'.")
                    self._advance()
                    self._advance()
                    ns_token = self._consume('IDENTIFIER', "Expected identifier after 'using namespace'")
                    self._consume('PUNCTUATOR', "Expected ';' after 'using namespace'", value=';')
                    if ns_token.value == 'std':
                        self.std_namespace_is_active = True
                        logging.info(
                            f"[parse_program] 'using namespace std;' processed. Flag set.")
                    else:
                        logging.info(f"[parse_program] 'using namespace {ns_token.value};' (non-std) processed.")
                    continue
                elif self._check_type_start():
                    logging.debug("[parse_program] Type start detected.")
                    decl = self._parse_external_declaration()
                    if decl: declarations.append(decl)
                    if self.current_token == start_while_token and decl is None: logging.warning(
                        f"[parse_program] _parse_external_declaration returned None but token did not advance. Current: {self.current_token}.")
                elif self._match('PUNCTUATOR', ';'):
                    logging.info("[parse_program] Empty top-level statement ignored.")
                    continue
                else:
                    self._error(f"Unexpected token at top level. Expected type specifier, 'using', or ';'")
            except ParseError as e:
                logging.error(f"[parse_program] ParseError: {e}")
                if self.current_token == start_while_token and self.current_token and self.current_token.type != "EOF": logging.warning(
                    f"[parse_program] Parser stuck on {self.current_token} after error. Forcing advance."); self._advance();
                if not self.current_token or self.current_token.type == "EOF": break
                logging.info("[parse_program] Attempting synchronization...")
                self._synchronize(context='top_level')
                if not self.current_token or self.current_token.type == "EOF": break
                logging.info(f"[parse_program] Resynchronized at {self.current_token}.")
            if self.current_token == start_while_token and self.current_token.type != "EOF": logging.critical(
                f"[parse_program] Parser did not advance. Stuck on {self.current_token}. Breaking."); self._advance(); break
        if parse_count >= max_parse_attempts: logging.error(
            f"[parse_program] Max parse attempts ({max_parse_attempts}) reached.")
        line = prog_start_token.original_line if prog_start_token else 1
        col = prog_start_token.column if prog_start_token else 1
        return Program(declarations, line=line, column=col)

    def _synchronize(self, context='unknown'):
        skipped_tokens_log = []
        logging.debug(f"[_synchronize] Entering synchronization (context: {context}). Current: {self.current_token}")
        phrase_terminators = {';', '}'}
        next_phrase_starters = {'{', 'if', 'while', 'for', 'do', 'return', 'switch', 'case', 'default', 'break',
                                'continue', 'goto', 'try', 'catch', 'throw', 'void', 'char', 'int', 'long', 'float',
                                'double', 'const', 'static', 'extern', 'struct', 'class', 'enum', 'union', 'typedef',
                                'using', 'namespace', 'template', 'public', 'private', 'protected', 'string'}
        while self.current_token and self.current_token.type != "EOF":
            token = self.current_token
            token_str = f"<{token.type},{repr(token.value)} L{token.original_line}C{token.column}>"
            if token.type == 'PUNCTUATOR' and token.value in phrase_terminators:
                skipped_tokens_log.append(token_str)
                if token.value == ';':
                    self._advance()
                    logging.info(
                        f"[_synchronize] Recovered by consuming until ';'. Skipped: {' '.join(skipped_tokens_log)}")
                else:
                    logging.info(
                        f"[_synchronize] Recovered by finding '}}'. Will resume before it. Skipped: {' '.join(skipped_tokens_log)}")
                return
            if self._check_type_start(): logging.info(
                f"[_synchronize] Resuming before potential type '{token.value}'. Skipped: {' '.join(skipped_tokens_log)}"); return
            if token.type == 'KEYWORD' and token.value in next_phrase_starters: logging.info(
                f"[_synchronize] Resuming before keyword '{token.value}'. Skipped: {' '.join(skipped_tokens_log)}"); return
            if token.type == 'PUNCTUATOR' and token.value == '{': logging.info(
                f"[_synchronize] Resuming before '{{'. Skipped: {' '.join(skipped_tokens_log)}"); return
            skipped_tokens_log.append(token_str)
            logging.debug(f"[_synchronize] Skipping token: {token}")
            self._advance()
        logging.info(f"[_synchronize] Reached EOF while synchronizing. Skipped: {' '.join(skipped_tokens_log)}")

    # --- 函数与变量声明解析 ---
    def _parse_external_declaration(self):
        start_token = self.current_token
        logging.debug(f"[_parse_external_declaration] Starting. Current token: {start_token}")
        decl_type = self._parse_type()
        if decl_type is None: return None
        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        full_type_str = decl_type + '*' * pointer_level
        logging.debug(f"[_parse_external_declaration] Parsed full type: '{full_type_str}'")
        if not self._check('IDENTIFIER'): self._error(
            f"Expected identifier after type specifier '{full_type_str}', but found {self.current_token}",
            token=self.current_token); return None
        name_token = self._consume('IDENTIFIER', "Expected identifier after type specifier")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        logging.debug(
            f"[_parse_external_declaration] Parsed identifier: '{name_identifier.name}'. Next token: {self.current_token}")
        line_info = start_token or name_token
        start_line = line_info.original_line if line_info else None
        start_col = line_info.column if line_info else None
        if self._check('PUNCTUATOR', '('):
            logging.debug(
                f"[_parse_external_declaration] Found '(', parsing as function for '{name_identifier.name}'.")
            self._advance()  # Consume '('
            params = self._parse_parameter_list()  # Use the REVISED parameter list parsing
            # _parse_parameter_list now consumes the final ')' or errors
            if self._check('PUNCTUATOR', '{'):
                logging.debug(
                    f"[_parse_external_declaration] Parsing function definition body for {name_identifier.name}")
                body = self._parse_compound_statement()
                return FunctionDefinition(full_type_str, name_identifier, params, body, line=start_line,
                                          column=start_col)
            elif self._match('PUNCTUATOR', ';'):
                logging.info(
                    f"[_parse_external_declaration] Parsed function prototype: {full_type_str} {name_identifier.name}(...)")
                decl_node = DeclarationStatement(full_type_str, name_identifier, None, line=start_line,
                                                 column=start_col)
                setattr(decl_node, 'is_prototype', True)
                setattr(decl_node, 'prototype_params', params)
                return decl_node
            else:
                self._error("Expected '{' for function body or ';' for prototype after parameter list ')'")
                return None
        else:
            logging.debug(
                f"[_parse_external_declaration] Did not find '(', parsing as variable for '{name_identifier.name}'.")
            initializer = None
            if self._match('OPERATOR', '='): initializer = self._parse_assignment_expression()
            self._consume('PUNCTUATOR', "Expected ';' after global variable declaration", value=';')
            decl_node = DeclarationStatement(full_type_str, name_identifier, initializer, line=start_line,
                                             column=start_col)
            setattr(decl_node, 'is_prototype', False)
            return decl_node

    def _parse_parameter_list(self):
        """解析函数参数列表。假定 '(' 已被调用者消耗。"""
        start_token = self.current_token
        logging.debug(f"[_parse_parameter_list] Starting. Current: {start_token}")
        params = []

        # 处理特殊情况: () 空列表
        if self._check('PUNCTUATOR', ')'):
            self._consume('PUNCTUATOR', "Expected ')' for empty parameter list", value=')')
            logging.debug(f"[_parse_parameter_list] Parsed empty list '()'.")
            return []

        # 处理特殊情况: (void)
        if self._check('KEYWORD', 'void') and self._check_peek('PUNCTUATOR', ')'):
            self._advance()  # void
            self._consume('PUNCTUATOR', "Expected ')' after 'void'", value=')')
            logging.debug(f"[_parse_parameter_list] Parsed '(void)'.")
            return []

        # --- 解析参数循环 ---
        while True:
            if self._check('PUNCTUATOR', ')'):  # 如果遇到')',说明列表结束（或可能是错误后）
                break

            # 解析一个参数 (type [*] [name])
            param_node = self._parse_parameter()
            if param_node is None:
                # _parse_parameter 应该已经报错并抛出异常
                # 如果没有（例如未来修改导致返回None），这里需要处理
                self._error("Failed to parse parameter", self.current_token or start_token)
                # 尝试同步以继续查找可能的错误
                self._synchronize(context='parameter_list')
                continue  # 同步后尝试继续解析下一个参数或找到 ')'

            params.append(param_node)
            logging.debug(f"[_parse_parameter_list] Parsed one parameter: {param_node}")

            # 查看参数后面是什么：逗号 or 右括号
            if self._check('PUNCTUATOR', ','):
                logging.debug(f"[_parse_parameter_list] Found comma, consuming and looking for next param or ...")
                self._consume('PUNCTUATOR', "Expected ','", value=',')  # 消耗逗号

                # 逗号后检查 '...'
                if self._match('OPERATOR', '...'):
                    logging.info("Parsed varargs '...'. Parameter list must end here.")
                    params.append(Parameter("...", None, line=self._last_consumed_token_for_error.original_line,
                                            column=self._last_consumed_token_for_error.column))
                    if not self._check('PUNCTUATOR', ')'):
                        # Varargs后面必须是')'
                        self._error("Expected ')' after varargs '...'")
                    break  # 结束循环，准备消耗 ')'

                # 逗号后检查是否意外地是 ')' (拖尾逗号)
                if self._check('PUNCTUATOR', ')'):
                    self._error("Trailing comma not allowed in parameter list",
                                token=self._last_consumed_token_for_error)  # 定位在逗号
                    break
                # 如果逗号后既不是 '...' 也不是 ')'，那么预期是下一个参数，继续循环
                logging.debug(f"[_parse_parameter_list] Continuing loop after comma.")

            elif self._check('PUNCTUATOR', ')'):
                # 参数后直接是 ')'，列表正常结束
                logging.debug(f"[_parse_parameter_list] Found ')' after parameter. Ending list.")
                break
            else:
                # 参数后既不是 ',' 也不是 ')'，语法错误
                self._error("Expected ',' or ')' after parameter", token=self.current_token)
                break  # 停止解析

        # --- 循环结束 ---
        # 消耗最后的 ')'
        self._consume('PUNCTUATOR', "Expected ')' to end parameter list", value=')')
        logging.debug(f"[_parse_parameter_list] Parsed parameters successfully: {params}")
        return params

    def _parse_parameter(self):
        """解析单个参数 (type [*] [name])。"""
        start_token = self.current_token
        logging.debug(f"[_parse_parameter] Starting. Current token: {start_token}")
        param_type_str = self._parse_type()
        if param_type_str is None: return None

        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        full_param_type = param_type_str + '*' * pointer_level

        name_id = None
        if self._check('IDENTIFIER'):
            name_token = self._consume('IDENTIFIER', "Expected parameter name")
            name_id = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
            logging.debug(f"[_parse_parameter] Parsed parameter name: {name_id.name}")
        else:
            logging.debug(f"[_parse_parameter] Parsed nameless parameter of type '{full_param_type}'.")

        line = start_token.original_line if start_token else None
        col = start_token.column if start_token else None
        return Parameter(full_param_type, name_id, line=line, column=col)

    def _parse_argument_list(self):
        """解析函数调用参数列表。假定 '(' 已被消耗。"""
        start_token = self.current_token
        logging.debug(f"[_parse_argument_list] Starting. Current: {start_token}")
        args = []

        # 处理空参数列表 ()
        if self._check('PUNCTUATOR', ')'):
            # 由调用者 (_parse_postfix_expression) 消耗 ')'
            logging.debug("[_parse_argument_list] Parsed empty argument list '()'.")
            return args

        # 解析一个或多个参数
        while True:
            # 解析一个参数表达式 (赋值表达式)
            logging.debug(f"[_parse_argument_list] Parsing argument expression. Current: {self.current_token}")
            arg_expr = self._parse_assignment_expression()
            if arg_expr is None:
                # 通常意味着语法错误
                self._error("Failed to parse function argument expression", self.current_token or start_token)
                return []  # 或抛出

            args.append(arg_expr)
            logging.debug(f"[_parse_argument_list] Parsed one argument: {arg_expr}. Next token: {self.current_token}")

            # 检查参数后面是逗号还是右括号
            if self._check('PUNCTUATOR', ','):
                # 找到逗号，消耗它，准备解析下一个参数
                self._consume('PUNCTUATOR', "Expected ','", value=',')  # 消耗逗号
                logging.debug(
                    f"[_parse_argument_list] Consumed comma. Looking for next argument or ')'. Current: {self.current_token}")
                # 检查是否是拖尾逗号 (后面直接是 ')')
                if self._check('PUNCTUATOR', ')'):
                    self._error("Unexpected ')' after comma in argument list (trailing comma)",
                                self._last_consumed_token_for_error)  # Error at comma
                    break  # 停止解析
                # 逗号后面不是 ')'，继续循环解析下一个参数
            elif self._check('PUNCTUATOR', ')'):
                # 找到右括号，参数列表正常结束
                logging.debug(f"[_parse_argument_list] Found ')' after argument. Ending list.")
                break
            else:
                # 参数后面既不是逗号也不是右括号，语法错误
                # <<<< 之前的错误发生在这里，是因为 _check(',') 返回了 False？>>>>
                # Let's log the token if this error occurs again
                logging.error(
                    f"[_parse_argument_list] After parsing argument, expected ',' or ')' but found: {self.current_token}")
                self._error("Expected ',' or ')' after function argument", self.current_token)
                break  # 停止解析

        # 注意：此函数不消耗最后的 ')'，由调用者 (_parse_postfix_expression) 负责
        logging.debug(f"[_parse_argument_list] Parsed arguments successfully: {args}")
        return args

    def _parse_statement(self):  # ... (same as previous version) ...
        loc_token = self.current_token
        line = loc_token.original_line if loc_token else None
        col = loc_token.column if loc_token else None
        logging.debug(f"[_parse_statement] Current token: {self.current_token}")
        if self._check_type_start(): return self._parse_declaration_statement()
        if self._check('PUNCTUATOR', '{'): return self._parse_compound_statement()
        if self._match('KEYWORD', 'if'): return self._parse_if_statement(line, col)
        if self._match('KEYWORD', 'while'): return self._parse_while_statement(line, col)
        if self._match('KEYWORD', 'for'): return self._parse_for_statement(line, col)
        if self._match('KEYWORD', 'do'): return self._parse_do_while_statement(line, col)
        if self._match('KEYWORD', 'return'): return self._parse_return_statement(line, col)
        if self._match('KEYWORD', 'break'): self._consume('PUNCTUATOR', "Expected ';'",
                                                          value=';'); return BreakStatement(line=line, column=col)
        if self._match('KEYWORD', 'continue'): self._consume('PUNCTUATOR', "Expected ';'",
                                                             value=';'); return ContinueStatement(line=line, column=col)
        if self._match('PUNCTUATOR', ';'): logging.debug("Parsed empty statement (';')."); return None
        if not self.current_token or self.current_token.type == 'EOF' or self._check('PUNCTUATOR', '}'): self._error(
            "Expected statement or expression, found end of block or file", token=loc_token); return None
        expr = self._parse_expression()
        if expr is None: self._error("Invalid or missing expression where statement expected",
                                     token=loc_token); return None
        self._consume('PUNCTUATOR', "Expected ';' after expression statement", value=';')
        return ExpressionStatement(expr, line=line, column=col)

    def _parse_compound_statement(self):
        start_token = self.current_token
        start_line = start_token.original_line if start_token else None
        start_col = start_token.column if start_token else None
        logging.debug(f"[_parse_compound_statement] Starting with token: {start_token}")
        self._consume('PUNCTUATOR', "Expected '{' to start compound statement", value='{')
        statements = []
        loop_guard = 0
        max_loop_guard = 1000
        while not self._check('PUNCTUATOR', '}'):
            loop_guard += 1
            if loop_guard > max_loop_guard: self._error("Parser likely stuck in compound statement loop",
                                                        token=self.current_token or start_token); break
            if not self.current_token or self.current_token.type == 'EOF': self._error(
                "Unexpected end of input, expected '}' to close block", token=start_token); break
            token_before_stmt_parse = self.current_token
            try:
                stmt = self._parse_statement()
                if stmt is not None: statements.append(stmt)
                if self.current_token == token_before_stmt_parse and not self._check('PUNCTUATOR', '}'):
                    logging.warning(
                        f"[_parse_compound_statement] Token did not advance. Current: {self.current_token}.")
                    self._error("Parser stuck inside compound statement", self.current_token)  # Trigger recovery
            except ParseError as inner_e:
                logging.error(f"[_parse_compound_statement] ParseError within block: {inner_e}")
                logging.info("[_parse_compound_statement] Attempting recovery within block...")
                self._synchronize(context='block_statement')
        self._consume('PUNCTUATOR', "Expected '}' to end compound statement", value='}')
        return CompoundStatement(statements, line=start_line, column=start_col)

    def _parse_declaration_statement(self):
        start_token = self.current_token
        logging.debug(f"[_parse_declaration_statement] Starting. Current token: {start_token}")
        decl_type = self._parse_type()
        if decl_type is None: return None
        pointer_level = 0
        while self._match('OPERATOR', '*'): pointer_level += 1
        full_type_str = decl_type + '*' * pointer_level
        if not self._check('IDENTIFIER'): self._error(
            f"Expected variable name after type '{full_type_str}', found {self.current_token}",
            token=self.current_token); return None
        name_token = self._consume('IDENTIFIER', f"Expected variable name after type '{full_type_str}'")
        name_identifier = Identifier(name_token.value, line=name_token.original_line, column=name_token.column)
        initializer = None
        if self._match('OPERATOR', '='):
            initializer = self._parse_assignment_expression()
            if initializer is None: self._error("Invalid initializer expression after '='", token=self.current_token)
        elif self._check('PUNCTUATOR', '{'):
            logging.debug("Parsing brace initializer (simplified).")
            self._consume('PUNCTUATOR', "Expected '{'", value='{')
            init_expr_in_brace = self._parse_assignment_expression()  # Simplification
            if init_expr_in_brace is None: self._error("Expected expression inside {}", token=self.current_token)
            self._consume('PUNCTUATOR', "Expected '}'", value='}')
            initializer = init_expr_in_brace
        self._consume('PUNCTUATOR', "Expected ';'", value=';')
        line_info = start_token or name_token
        decl_line = line_info.original_line if line_info else None
        decl_col = line_info.column if line_info else None
        decl_node = DeclarationStatement(full_type_str, name_identifier, initializer, line=decl_line, column=decl_col)
        setattr(decl_node, 'is_prototype', False)
        return decl_node

    def _parse_if_statement(self, line, col):
        logging.debug(f"[_parse_if_statement] Starting.")
        self._consume('PUNCTUATOR', "Expected '(' after 'if'", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'if'"); return None
        self._consume('PUNCTUATOR', "Expected ')' after 'if' condition", value=')')
        then_branch = self._parse_statement()
        else_branch = None
        if self._match('KEYWORD', 'else'): else_branch = self._parse_statement()
        return IfStatement(condition, then_branch, else_branch, line=line, column=col)

    def _parse_while_statement(self, line, col):
        logging.debug(f"[_parse_while_statement] Starting.")
        self._consume('PUNCTUATOR', "Expected '(' after 'while'", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'while'"); return None
        self._consume('PUNCTUATOR', "Expected ')' after 'while' condition", value=')')
        body = self._parse_statement()
        return WhileStatement(condition, body, line=line, column=col)

    def _parse_for_statement(self, line, col):
        logging.debug(f"[_parse_for_statement] Starting.")
        self._consume('PUNCTUATOR', "Expected '(' after 'for'", value='(')
        init = None
        if not self._check('PUNCTUATOR', ';'):
            if self._check_type_start():
                decl_start_token = self.current_token
                decl_type_str = self._parse_type()
                if not decl_type_str: self._error("Invalid type in 'for' init", decl_start_token); return None
                ptr_level = 0
                while self._match('OPERATOR', '*'): ptr_level += 1; decl_type_str += '*' * ptr_level
                if not self._check("IDENTIFIER"): self._error("Expected identifier in 'for' declaration",
                                                              self.current_token); return None
                name_tok = self._consume("IDENTIFIER", "Identifier expected in for declaration")
                name_id = Identifier(name_tok.value, line=name_tok.original_line, column=name_tok.column)
                init_expr = None
                if self._match('OPERATOR', '='): init_expr = self._parse_assignment_expression();
                if init_expr is None and self._last_consumed_token_for_error.value == '=': self._error(
                    "Invalid initializer in 'for' declaration", self.current_token); return None
                init_line = decl_start_token.original_line if decl_start_token else None
                init_col = decl_start_token.column if decl_start_token else None
                init = DeclarationStatement(decl_type_str, name_id, init_expr, line=init_line, column=init_col)
                setattr(init, 'is_for_init', True)
            else:
                init = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after 'for' initializer", value=';')
        condition = None
        if not self._check('PUNCTUATOR', ';'): condition = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ';' after 'for' condition", value=';')
        update = None
        if not self._check('PUNCTUATOR', ')'): update = self._parse_expression()
        self._consume('PUNCTUATOR', "Expected ')' after 'for' clauses", value=')')
        body = self._parse_statement()
        return ForStatement(init, condition, update, body, line=line, column=col)

    def _parse_do_while_statement(self, line, col):
        logging.debug(f"[_parse_do_while_statement] Starting.")
        body = self._parse_statement()
        if not self._match('KEYWORD', 'while'): self._error("Expected 'while' after 'do' body",
                                                            token=self.current_token or self._last_consumed_token_for_error); return None
        self._consume('PUNCTUATOR', "Expected '(' after 'while'", value='(')
        condition = self._parse_expression()
        if condition is None: self._error("Missing condition in 'do-while'"); return None
        self._consume('PUNCTUATOR', "Expected ')' after 'do-while' condition", value=')')
        self._consume('PUNCTUATOR', "Expected ';' after 'do-while'", value=';')
        return DoWhileStatement(body, condition, line=line, column=col)

    def _parse_return_statement(self, line, col):
        logging.debug(f"[_parse_return_statement] Starting.")
        value = None
        if not self._check('PUNCTUATOR', ';'):
            value = self._parse_expression()
            if value is None and not self._check('PUNCTUATOR', ';'): self._error("Invalid expression for 'return'",
                                                                                 self.current_token)
        self._consume('PUNCTUATOR', "Expected ';' after return", value=';')
        return ReturnStatement(value, line=line, column=col)

    def _get_precedence(self, token):
        if not token: return -1
        if token.type == 'OPERATOR' or (token.type == 'PUNCTUATOR' and token.value == '?'): return self.PRECEDENCE.get(
            token.value, -1)
        return -1

    def _parse_expression(self, min_precedence=0):
        lhs = self._parse_unary_expression()
        if lhs is None: return None
        while True:
            op_token = self.current_token
            is_binary_op_candidate = op_token and (
                    op_token.type == 'OPERATOR' or (op_token.type == 'PUNCTUATOR' and op_token.value == '?'))
            if not is_binary_op_candidate: break
            precedence = self._get_precedence(op_token)
            if precedence < min_precedence: break
            if op_token.value == '?':
                self._advance()
                true_expr = self._parse_expression(0)
                if true_expr is None: self._error("Expected expression after '?'", token=op_token); return None
                self._consume('PUNCTUATOR', "Expected ':' in ternary op", value=':')
                false_expr = self._parse_expression(self.PRECEDENCE['?'])
                if false_expr is None: self._error("Expected expression after ':'",
                                                   token=self.current_token); return None
                op_line = op_token.original_line
                op_col = op_token.column
                lhs = TernaryOp(lhs, true_expr, false_expr, line=op_line, column=op_col)
                continue
            next_min_precedence = precedence + (1 if op_token.value not in self.RIGHT_ASSOC else 0)
            self._advance()
            rhs = self._parse_expression(next_min_precedence)
            if rhs is None: self._error(f"Expected expression after binary op '{op_token.value}'",
                                        token=op_token); return None
            op_line = op_token.original_line
            op_col = op_token.column
            is_lvalue_syntax_ok = isinstance(lhs, (Identifier, ArraySubscript, MemberAccess)) or (
                    isinstance(lhs, UnaryOp) and lhs.op == '*')
            if op_token.value in self.RIGHT_ASSOC and op_token.value != '?' and not is_lvalue_syntax_ok: self._error(
                f"Invalid L-value for assignment '{op_token.value}'", token=op_token)
            lhs = BinaryOp(op_token.value, lhs, rhs, line=op_line, column=op_col)
        return lhs

    def _parse_assignment_expression(self):
        return self._parse_expression(min_precedence=1)

    def _parse_unary_expression(self):
        op_token = self.current_token
        prefix_ops = ['-', '+', '!', '~', '++', '--', '*', '&']
        if self._check('OPERATOR') and op_token.value in prefix_ops:
            op_str = op_token.value
            op_line = op_token.original_line if op_token else None
            op_col = op_token.column if op_token else None
            self._advance()
            ast_op = '++p' if op_str == '++' else '--p' if op_str == '--' else op_str
            operand = self._parse_unary_expression()
            if operand is None: self._error(f"Expected expression after unary op '{op_str}'",
                                            token=op_token); return None
            is_operand_lvalue = isinstance(operand, (Identifier, ArraySubscript, MemberAccess)) or (
                    isinstance(operand, UnaryOp) and operand.op == '*')
            if op_str in ['++', '--'] and not is_operand_lvalue: self._error(
                f"Operand of prefix op '{op_str}' must be L-value", token=op_token)
            if op_str == '&' and not (is_operand_lvalue or isinstance(operand, Identifier)): pass  # Simplified check
            return UnaryOp(ast_op, operand, line=op_line, column=op_col)
        elif self._match('KEYWORD', 'sizeof'):
            start_sizeof_token = self._last_consumed_token_for_error
            sizeof_line = start_sizeof_token.original_line if start_sizeof_token else None
            sizeof_col = start_sizeof_token.column if start_sizeof_token else None
            target_operand_for_ast = None
            if self._match('PUNCTUATOR', '('):
                if self._check_type_start():
                    parsed_target_type = self._parse_type()
                    if parsed_target_type:
                        ptr_level = 0
                        while self._match('OPERATOR',
                                          '*'): ptr_level += 1; target_operand_for_ast = parsed_target_type + "*" * ptr_level
                    else:
                        self._error("Invalid type specifier in sizeof(...)", token=start_sizeof_token)
                        return None
                else:
                    target_operand_for_ast = self._parse_expression()
                if target_operand_for_ast is None: self._error("Invalid expression in sizeof(...)",
                                                               token=self.current_token or start_sizeof_token); return None
                self._consume('PUNCTUATOR', "Expected ')' after sizeof operand", value=')')
            else:
                target_operand_for_ast = self._parse_unary_expression()
            if target_operand_for_ast is None: self._error("Expected expression or type after 'sizeof'",
                                                           token=start_sizeof_token); return None
            return UnaryOp('sizeof', target_operand_for_ast, line=sizeof_line, column=sizeof_col)
        else:
            return self._parse_postfix_expression()

    def _parse_postfix_expression(self):
        expr = self._parse_primary_expression()
        if expr is None: return None

        while True:
            op_token = self.current_token
            if not op_token or op_token.type == "EOF": break

            op_line = op_token.original_line if op_token else (getattr(expr, 'line', None))
            op_col = op_token.column if op_token else (getattr(expr, 'column', None))

            if self._match('PUNCTUATOR', '('):
                args = self._parse_argument_list()  # Corrected arg list parsing
                self._consume('PUNCTUATOR', "Expected ')' after function call arguments", value=')')
                expr = CallExpression(expr, args, line=op_line, column=op_col)
                continue  # Continue checking for more postfix ops

            elif self._match('PUNCTUATOR', '['):
                index_expr = self._parse_expression()
                # --- Correct indentation START ---
                if index_expr is None:
                    self._error("Expected index expression inside '[]'", token=op_token)
                    break  # Stop postfix parsing on error
                self._consume('PUNCTUATOR', "Expected ']' after array index", value=']')
                expr = ArraySubscript(expr, index_expr, line=op_line, column=op_col)
                # --- Correct indentation END ---
                continue  # Continue checking

            elif self._match('OPERATOR', '++'):
                is_expr_lvalue = isinstance(expr, (Identifier, ArraySubscript, MemberAccess)) or \
                                 (isinstance(expr, UnaryOp) and expr.op == '*')
                if not is_expr_lvalue:
                    self._error("Operand of postfix '++' must be an lvalue", token=self._last_consumed_token_for_error)
                expr = UnaryOp('p++', expr, line=op_line, column=op_col)  # Create node regardless
                continue  # Continue checking

            elif self._match('OPERATOR', '--'):
                is_expr_lvalue = isinstance(expr, (Identifier, ArraySubscript, MemberAccess)) or \
                                 (isinstance(expr, UnaryOp) and expr.op == '*')
                if not is_expr_lvalue:
                    self._error("Operand of postfix '--' must be an lvalue", token=self._last_consumed_token_for_error)
                expr = UnaryOp('p--', expr, line=op_line, column=op_col)  # Create node regardless
                continue  # Continue checking

            elif self._match('OPERATOR', '.'):
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '.'")
                member_id = Identifier(member_token.value, line=member_token.original_line, column=member_token.column)
                expr = MemberAccess(expr, member_id, is_pointer=False, line=op_line, column=op_col)
                continue  # Continue checking

            elif self._match('OPERATOR', '->'):
                member_token = self._consume('IDENTIFIER', "Expected member identifier after '->'")
                member_id = Identifier(member_token.value, line=member_token.original_line, column=member_token.column)
                expr = MemberAccess(expr, member_id, is_pointer=True, line=op_line, column=op_col)
                continue  # Continue checking
            else:
                break  # No more postfix operators matched
        return expr


    def _parse_primary_expression(self):
        token = self.current_token
        if not token or token.type == "EOF": self._error("Unexpected EOF expecting primary expression",
                                                         token=token); return None
        line = token.original_line if token else None
        col = token.column if token else None
        node = None

        # Literals and Identifiers
        if self._match('INTEGER_LITERAL'):
            node = IntegerLiteral(self._last_consumed_token_for_error.value, line=line, column=col)
        elif self._match('FLOAT_LITERAL'):
            node = FloatLiteral(self._last_consumed_token_for_error.value, line=line, column=col)
        elif self._match('STRING_LITERAL'):
            node = StringLiteral(self._last_consumed_token_for_error.value, line=line, column=col)
        elif self._match('CHAR_LITERAL'):
            node = CharLiteral(self._last_consumed_token_for_error.value, line=line, column=col)
        elif self._match('IDENTIFIER'):
            node = Identifier(self._last_consumed_token_for_error.value, line=line, column=col)
        elif self._match('KEYWORD', 'true'):
            node = BooleanLiteral(True, line=line, column=col)
        elif self._match('KEYWORD', 'false'):
            node = BooleanLiteral(False, line=line, column=col)
        elif self._match('KEYWORD', 'nullptr'):
            node = NullPtrLiteral(line=line, column=col)

        # Parenthesized expression or C-style cast
        elif self._check('PUNCTUATOR', '('):
            paren_start_token = self.current_token
            paren_line = paren_start_token.original_line
            paren_col = paren_start_token.column
            self._advance()  # Consume '('
            is_cast = False
            final_cast_type = None
            if self._check_type_start():
                cast_target_type_str = self._parse_type()
                if cast_target_type_str:
                    pointer_level = 0
                    while self._match('OPERATOR', '*'): pointer_level += 1
                    final_cast_type = cast_target_type_str + "*" * pointer_level
                    if self._check('PUNCTUATOR', ')'):
                        self._consume('PUNCTUATOR', "Expected ')' after type in cast", value=')')
                        expression_to_cast = self._parse_unary_expression()  # Parse expression being cast
                        if expression_to_cast is None:
                            self._error("Expected expression following C-style cast '(type)'",
                                        token=self.current_token or paren_start_token)
                            return None
                        # Now it's safe to use final_cast_type
                        node = CastExpression(final_cast_type, expression_to_cast, line=paren_line, column=paren_col)
                        is_cast = True  # Mark as successful cast
                    else:  # Missing ')' after potential type - error
                        # final_cast_type is assigned, but syntax is wrong
                        self._error(
                            f"Expected ')' to complete cast after type '{final_cast_type}', found {self.current_token}",
                            token=self.current_token)
                        return None
            if not is_cast:
                node = self._parse_expression()  # Parse as grouped expression
                if node is None: self._error("Expected expression inside parentheses",
                                             token=self.current_token or paren_start_token); return None
                self._consume('PUNCTUATOR', "Expected ')' to close parenthesized expression", value=')')
        else:
            self._error(f"Unexpected token, expected primary expression", token=token)
            return None
        return node



def print_ast_tree(node, indent="", last=True, prefix=""):
    if node is None: return
    connector = '└── ' if last else '├── '
    node_repr = ""
    children_info = []
    line = getattr(node, 'line', None)
    col = getattr(node, 'column', None)
    loc_info = f" (L{line}:C{col})" if line is not None and col is not None else f" (L{line})" if line is not None else " (NoLoc)"
    node_type_name = type(node).__name__
    specific_info = ""
    if isinstance(node, Program):
        children_info = [("declarations", node.declarations)]
    elif isinstance(node, FunctionDefinition):
        specific_info = f": {node.name.name} (returns: {node.return_type})"
        children_info = [("name_id", node.name),
                         ("params", node.params),
                         ("body", node.body)]
    elif isinstance(node, Parameter):
        name_str = node.name.name if node.name else "<unnamed>"
        specific_info = f": {name_str} (type: {node.param_type})"
    elif isinstance(node, CompoundStatement):
        children_info = [("statements", node.statements)]
    elif isinstance(node, DeclarationStatement):
        proto_str = " (prototype)" if getattr(node, 'is_prototype',
                                              False) else ""
        for_init_str = " (for-init)" if getattr(node,
                                                'is_for_init',
                                                False) else ""
        specific_info = f": {node.name.name} (type: {node.decl_type}){proto_str}{for_init_str}"
        children_info = [
            ("name_id", node.name), ("initializer", node.initializer)]
    elif isinstance(node, ExpressionStatement):
        children_info = [("expression", node.expression)]
    elif isinstance(node, IfStatement):
        children_info = [("condition", node.condition), ("then_branch", node.then_branch),
                         ("else_branch", node.else_branch)]
    elif isinstance(node, WhileStatement):
        children_info = [("condition", node.condition), ("body", node.body)]
    elif isinstance(node, ForStatement):
        children_info = [("init", node.init), ("condition", node.condition), ("update", node.update),
                         ("body", node.body)]
    elif isinstance(node, DoWhileStatement):
        children_info = [("body", node.body), ("condition", node.condition)]
    elif isinstance(node, BreakStatement):
        pass
    elif isinstance(node, ContinueStatement):
        pass
    elif isinstance(node, ReturnStatement):
        children_info = [("value", node.value)]
    elif isinstance(node, Identifier):
        specific_info = f": {node.name}"
    elif isinstance(node, IntegerLiteral):
        specific_info = f": {node.value}"
    elif isinstance(node, FloatLiteral):
        specific_info = f": {node.value}"
    elif isinstance(node, StringLiteral):
        specific_info = f": {repr(node.value)}"
    elif isinstance(node, CharLiteral):
        specific_info = f": {repr(node.value)}"
    elif isinstance(node, BooleanLiteral):
        specific_info = f": {node.value}"
    elif isinstance(node, NullPtrLiteral):
        pass
    elif isinstance(node, BinaryOp):
        specific_info = f": '{node.op}'"
        children_info = [("left", node.left), ("right", node.right)]
    elif isinstance(node, UnaryOp):
        op_display = node.op
        operand_val = node.operand
        is_sizeof_type = op_display == 'sizeof' and isinstance(
            operand_val,
            str)
        op_prefix = "type" if is_sizeof_type else "operand"
        specific_info = f": '{op_display}'"
        children_info = [
            (op_prefix, operand_val)]
    elif isinstance(node, CallExpression):
        children_info = [("function_expr", node.function), ("arguments", node.args)]
    elif isinstance(node, ArraySubscript):
        children_info = [("array_expr", node.array_expression), ("index_expr", node.index_expression)]
    elif isinstance(node, MemberAccess):
        op = '->' if node.is_pointer_access else '.'
        specific_info = f": {op}{node.member_identifier.name}"
        children_info = [
            ("object_expr", node.object_or_pointer_expression), ("member_id", node.member_identifier)]
    elif isinstance(node, CastExpression):
        specific_info = f": (to type='{node.target_type}')"
        children_info = [("expression", node.expression)]
    elif isinstance(node, TernaryOp):
        node_type_name = "TernaryOp"
        children_info = [("condition", node.condition),
                         ("true_expr", node.true_expression),
                         ("false_expr", node.false_expression)]
    elif isinstance(node, ASTNode):
        pass
    if hasattr(node, 'is_prototype') and node.is_prototype and hasattr(node, 'prototype_params'): children_info.append(
        ("prototype_params", node.prototype_params))
    node_repr = f"{node_type_name}{specific_info}{loc_info}"
    print(f"{indent}{connector}{prefix}{node_repr}")
    valid_children_to_print = [(name, child) for name, child in children_info if child is not None]
    child_count = len(valid_children_to_print)
    for i, (child_name, child_node_or_list) in enumerate(valid_children_to_print):
        is_last_child = (i == child_count - 1)
        current_child_prefix = f"{child_name}: " if child_name else ""
        new_indent = indent + ('    ' if is_last_child else '│   ')
        if isinstance(child_node_or_list, list):
            actual_list_items = [item for item in child_node_or_list if item is not None]
            num_actual_items = len(actual_list_items)
            list_label = f"[{num_actual_items} item(s)]" if num_actual_items > 0 else "[empty list]"
            print(f"{new_indent}{'└── ' if is_last_child else '├── '}{current_child_prefix}{list_label}")
            if num_actual_items > 0:
                list_items_indent = new_indent + ('    ' if is_last_child else '│   ')
                for j, item in enumerate(actual_list_items): is_last_item_in_list = (
                        j == num_actual_items - 1); item_prefix_in_list = f"[{j}]: "; print_ast_tree(item,
                                                                                                     indent=list_items_indent,
                                                                                                     last=is_last_item_in_list,
                                                                                                     prefix=item_prefix_in_list)
        elif isinstance(child_node_or_list, ASTNode):
            print_ast_tree(child_node_or_list, indent=new_indent, last=is_last_child, prefix=current_child_prefix)
        elif isinstance(child_node_or_list, str):
            print(f"{new_indent}{'└── ' if is_last_child else '├── '}{current_child_prefix}'{child_node_or_list}'")
        else:
            print(
                f"{new_indent}{'└── ' if is_last_child else '├── '}{current_child_prefix}UnknownChild: {repr(child_node_or_list)}")


if __name__ == "__main__":
    if len(sys.argv) != 2: print(f"用法: python parser.py <input_file.cpp>", file=sys.stderr); sys.exit(1)
    input_file_path = sys.argv[1]
    raw_code = None
    processed_code = None
    tokens = None
    ast = None
    line_map = {}
    had_errors = False
    is_effectively_empty = False
    try:
        print(f"--- Stage 1: Reading File ---")
        print(f"Reading file: {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            raw_code = infile.read()
        print("File reading complete.")
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.", file=sys.stderr)
        had_errors = True
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        had_errors = True
    if had_errors: sys.exit(1)
    try:
        print(f"\n--- Stage 2: Preprocessing ---")
        from preprocess import BasicPreprocessor

        print("Running preprocessor...")
        preprocessor = BasicPreprocessor()
        processed_code, line_map = preprocessor.process(raw_code)
        print("Preprocessing complete.")
    except ImportError:
        print(
            "Warning: preprocess.py not found. Skipping preprocessing stage.")
        processed_code = raw_code
        num_lines = raw_code.count(
            '\n') + 1
        line_map = {i: i for i in range(1, num_lines + 1)}
    except Exception as e:
        print(f"Error during preprocessing: {e}. Using raw code.",
              file=sys.stderr)
        processed_code = raw_code
        num_lines = raw_code.count('\n') + 1
        line_map = {i: i for i
                    in range(1,
                             num_lines + 1)}
    token_list = []
    if not had_errors and processed_code is not None:
        try:
            print(f"\n--- Stage 3: Lexical Analysis ---")
            print("Starting lexical analysis...")
            lexer = Lexer(processed_code, line_map)
            token_list = list(lexer.tokenize())
            print(f"Lexical analysis complete. Generated {len(token_list)} tokens.")
            logging.debug(f"First 50 tokens: {token_list[:50]}")
        except LexerError as e:
            print(e, file=sys.stderr)
            had_errors = True
        except Exception as e:
            print(f"Unexpected Lexer Error: {e}", file=sys.stderr)
            import \
                traceback

            traceback.print_exc()
            had_errors = True
    if not had_errors:
        is_effectively_empty = not token_list or (len(token_list) == 1 and token_list[0].type == 'EOF')
        if is_effectively_empty:
            print("\nInput is effectively empty. Creating empty Program AST.")
            ast = Program([])
        else:
            try:
                print(f"\n--- Stage 4: Syntax Analysis (Parsing) ---")
                print("Starting syntax analysis...")
                parser = Parser(iter(token_list))
                ast = parser.parse_program()
                print("\n--- Abstract Syntax Tree (Generated by Parser) ---")
                if ast:
                    print_ast_tree(ast)
                else:
                    if parser.current_token is None or parser.current_token.type == "EOF":
                        print("Parsing resulted in an empty AST.")
                    else:
                        print("AST generation failed.")
                        had_errors = True
                print("-------------------------------------------------")
                print("Syntax analysis complete.")
            except ParseError as final_e:
                print(f"\nSyntax analysis failed.", file=sys.stderr)
                had_errors = True
            except Exception as e:
                print(f"\nUnexpected error during syntax analysis: {e}", file=sys.stderr)
                import \
                    traceback

                traceback.print_exc()
                had_errors = True
    elif not had_errors and not token_list:
        print("\nLexer produced no tokens.")
        ast = Program([])
    print("\n--- Compilation Summary ---")
    if had_errors:
        print("Compilation failed due to errors.")
        sys.exit(1)
    else:
        print("Parsing completed successfully.")
        sys.exit(0)
