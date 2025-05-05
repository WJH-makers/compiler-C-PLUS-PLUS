# coding=utf-8
import logging
import re
import sys

from preprocess import BasicPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Token:
    def __init__(self, type, value, original_line, column):
        self.type = type
        self.value = value
        self.original_line = original_line
        self.column = column

    def __repr__(self):
        return f"<{self.type}, {repr(self.value)}, L{self.original_line}:C{self.column}>"


class LexerError(Exception):
    def __init__(self, message, original_line, column):
        super().__init__(f"Lexer Error at L{original_line}:C{column}: {message}")
        self.original_line = original_line
        self.column = column


class Lexer:
    def __init__(self, code, line_mapping):
        self.code = code
        self.position = 0
        self.processed_line = 1
        self.column = 1
        self.tokens = []
        self.line_mapping = line_mapping
        self.keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'inline', 'int', 'long', 'string', 'register', 'restrict', 'return', 'short',
            'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
            'unsigned', 'void', 'volatile', 'while',
            '_Bool', '_Complex', '_Imaginary',
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'bitand', 'bitor', 'bool',
            'catch', 'class', 'compl', 'const_cast', 'decltype', 'delete',
            'dynamic_cast', 'explicit', 'export', 'false', 'friend', 'mutable',
            'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator',
            'or', 'or_eq', 'private', 'protected', 'public', 'reinterpret_cast',
            'static_assert', 'static_cast', 'template', 'this', 'thread_local',
            'throw', 'true', 'try', 'typeid', 'typename', 'using', 'virtual',
            'wchar_t', 'xor', 'xor_eq',
        }
        self.token_specs = [
            ('COMMENT_MULTI', r'/\*.*?\*/'),
            ('COMMENT_SINGLE', r'//.*'),
            ('NEWLINE', r'\n'),
            ('WHITESPACE', r'[ \t\r\f\v]+'),
            ('RAW_STRING_LITERAL', r'R"((?P<delim>[^()\s]{0,16}))\((?P<content>.*?)\)\1"'),
            ('CHAR_LITERAL', r'(u8|[uUL])?\'([^\\\'\n]|\\([nrtv\\\'\"?abf0]|x[0-9a-fA-F]{1,2}|[0-7]{1,3}))\''),
            ('STRING_LITERAL', r'(u8|[uUL])?\"([^\\\"\n]|\\([nrtv\\\'\"?abf0]|x[0-9a-fA-F]{1,2}|[0-7]{1,3}))*\"'),
            ('OPERATOR_3CHAR', r'<<=|>>='),
            ('OPERATOR_ELLIPSIS', r'\.\.\.'),
            ('OPERATOR_2CHAR', r'->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|::'),
            ('OPERATOR_1CHAR', r'[+\-*/%&|^~!<>=?,.:]'),
            ('PUNCTUATOR', r'[{}();\[\]]'),
            ('BIN_INTEGER', r'0[bB][01]+(\'[01]+)*[uUlL]*'),
            ('HEX_INTEGER', r'0[xX][0-9a-fA-F]+(\'[0-9a-fA-F]+)*[uUlL]*'),
            ('FLOAT_LITERAL',
             r'(([0-9]+(\'[0-9]+)*\.[0-9]*(\'[0-9]+)*|\.[0-9]+(\'[0-9]+)*)([eE][+-]?[0-9]+(\'[0-9]+)*)?|[0-9]+(\'[0-9]+)*[eE][+-]?[0-9]+(\'[0-9]+)*)[fFlL]?'),
            ('OCT_INTEGER', r'0[0-7]+(\'[0-7]+)*[uUlL]*'),
            ('DEC_INTEGER', r'[0-9]+(\'[0-9]+)*[uUlL]*'),
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('MISMATCH', r'.')
        ]
        self.master_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.token_specs),
                                       re.DOTALL)

    def _interpret_escapes(self, s):
        escape_map = {'\\n': '\n', '\\t': '\t', '\\r': '\r', '\\\\': '\\', '\\\'': '\'', '\\"': '"', '\\a': '\a',
                      '\\b': '\b', '\\f': '\f', '\\v': '\v', '\\?': '?'}
        out = ""
        i = 0
        while i < len(s):
            if s[i] == '\\':
                if i + 1 < len(s):
                    esc2 = s[i:i + 2]
                    if esc2 in escape_map:
                        out += escape_map[esc2]
                        i += 2
                        continue
                    elif s[i + 1] >= '0' and s[i + 1] <= '7':
                        j = i + 1
                        while j < len(s) and s[j] >= '0' and s[j] <= '7' and (j - i - 1) < 3: j += 1
                        octal_val = int(s[i + 1:j], 8)
                        out += chr(octal_val)
                        i = j
                        continue
                    elif s[i + 1] in ('x', 'X') and i + 2 < len(s) and s[i + 2] in '0123456789abcdefABCDEF':
                        j = i + 2
                        while j < len(s) and s[j] in '0123456789abcdefABCDEF': j += 1
                        if j > i + 2:  # Check if at least one hex digit was found
                            hex_val = int(s[i + 2:j], 16)
                            out += chr(hex_val)
                            i = j
                            continue
            out += s[i]
            i += 1
        return out

    def _get_original_line(self, processed_line_num):
        original_line = self.line_mapping.get(processed_line_num)
        if original_line is None:
            logging.warning(f"No original line mapping for processed L{processed_line_num}.")
            return processed_line_num
        return original_line

    def tokenize(self):
        while self.position < len(self.code):
            match = self.master_regex.match(self.code, self.position)
            if not match:
                original_error_line = self._get_original_line(self.processed_line)
                raise LexerError(
                    f"Unable to match token starting near '{self.code[self.position:self.position + 10]}...'",
                    original_error_line, self.column)

            kind = match.lastgroup
            value = match.group(kind)
            start_column = self.column
            current_original_line = self._get_original_line(self.processed_line)

            # --- Skip whitespace and comments ---
            if kind in ['COMMENT_MULTI', 'COMMENT_SINGLE', 'NEWLINE', 'WHITESPACE']:
                lines_in_match = value.count('\n')
                if lines_in_match > 0:
                    self.processed_line += lines_in_match
                    self.column = len(value) - value.rfind('\n')
                else:
                    self.column += len(value)
                self.position = match.end()
                continue

            # --- Determine Token Type and Value ---
            token_type = None
            token_value = value  # Default value is the matched string

            if kind in ['OPERATOR_3CHAR', 'OPERATOR_2CHAR', 'OPERATOR_1CHAR', 'OPERATOR_ELLIPSIS']:
                token_type = 'OPERATOR'
            elif kind == 'PUNCTUATOR':
                token_type = 'PUNCTUATOR'
            elif kind == 'CHAR_LITERAL':
                token_type = 'CHAR_LITERAL'
                prefix_match = re.match(r'(u8|[uUL])?', token_value)
                prefix_len = prefix_match.end() if prefix_match else 0
                char_content = token_value[prefix_len + 1: -1]  # Extract content within ''
                token_value = self._interpret_escapes(char_content)  # Interpret escapes
            elif kind == 'STRING_LITERAL':
                token_type = 'STRING_LITERAL'
                prefix_match = re.match(r'(u8|[uUL])?', token_value)
                prefix_len = prefix_match.end() if prefix_match else 0
                string_content = token_value[prefix_len + 1: -1]  # Extract content within ""
                token_value = self._interpret_escapes(string_content)  # Interpret escapes
            elif kind == 'RAW_STRING_LITERAL':
                token_type = 'STRING_LITERAL'  # Treat raw string as regular string literal token type
                token_value = match.group('content')  # Value is just the content part
            elif kind in ['HEX_INTEGER', 'OCT_INTEGER', 'DEC_INTEGER', 'BIN_INTEGER']:
                token_type = 'INTEGER_LITERAL'
            elif kind == 'FLOAT_LITERAL':
                token_type = 'FLOAT_LITERAL'
            elif kind == 'IDENTIFIER':
                token_type = 'KEYWORD' if token_value in self.keywords else 'IDENTIFIER'
            elif kind == 'MISMATCH':
                raise LexerError(f"Illegal character encountered: '{token_value}'", current_original_line, start_column)
            else:
                raise LexerError(f"Unhandled token kind: {kind}", current_original_line, start_column)

            self.tokens.append(Token(token_type, token_value, current_original_line, start_column))
            lines_in_value = token_value.count('\n')
            if lines_in_value > 0:
                self.processed_line += lines_in_value
                self.column = len(token_value) - token_value.rfind('\n')
            else:
                self.column += len(match.group(kind))
            self.position = match.end()
        eof_original_line = self._get_original_line(self.processed_line)
        self.tokens.append(Token("EOF", "", eof_original_line, self.column))
        return self.tokens


# --- Main block for testing lexer ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python lexer.py <input_file.cpp>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    raw_code = None
    processed_code = None
    line_map = {}

    try:
        logging.info(f"正在读取文件: {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            raw_code = infile.read()
        try:
            logging.info("正在尝试运行预处理器...")
            preprocessor = BasicPreprocessor()
            processed_code, line_map = preprocessor.process(raw_code)
            logging.info("预处理完成.")
        except ImportError:
            logging.warning("警告: 找不到 preprocess.py。将对原始代码进行词法分析 (无行号映射)。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
        except Exception as e:
            logging.error(f"预处理时发生错误: {e}。将对原始代码进行词法分析 (无行号映射)。", exc_info=True)
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}

        if processed_code is not None:
            logging.info("正在进行词法分析...")
            lexer = Lexer(processed_code, line_map)
            tokens = lexer.tokenize()  # Returns list
            logging.info("词法分析完成.")
            print("\n词法分析结果 (Tokens):")
            for token in tokens:
                print(token)
        else:
            logging.error("错误：预处理后没有代码可供分析。")
            sys.exit(1)

    except FileNotFoundError:
        logging.error(f"错误: 文件 '{input_file_path}' 未找到")
        sys.exit(1)
    except LexerError as e:
        logging.error(e)
        sys.exit(1)
    except Exception as e:
        logging.error(f"发生未预料的错误: {e}", exc_info=True)
        sys.exit(1)
