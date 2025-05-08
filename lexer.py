# coding=utf-8
import logging
import re
import sys

# 尝试导入预处理器，如果失败则给出警告
try:
    from preprocess import BasicPreprocessor
except ImportError:
    BasicPreprocessor = None  # 定义为 None，稍后检查
    logging.warning("preprocess.py not found or BasicPreprocessor class missing. Preprocessing skipped.")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Token:
    def __init__(self, type, value, original_line, column):
        self.type = type
        self.value = value
        self.original_line = original_line
        self.column = column

    def __repr__(self):
        # 更健壮的 repr，处理 None 值
        type_repr = self.type or "NoneType"
        value_repr = repr(self.value) if self.value is not None else "NoneValue"
        line_repr = self.original_line if self.original_line is not None else "?"
        col_repr = self.column if self.column is not None else "?"
        return f"<{type_repr}, {value_repr}, L{line_repr}:C{col_repr}>"


class LexerError(Exception):
    def __init__(self, message, original_line, column):
        line_repr = original_line if original_line is not None else "?"
        col_repr = column if column is not None else "?"
        super().__init__(f"Lexer Error at L{line_repr}:C{col_repr}: {message}")
        self.original_line = original_line
        self.column = column


def _interpret_escapes(s):
    """解释字符串和字符字面量中的转义序列。"""
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
                elif '0' <= s[i + 1] <= '7':  # 八进制
                    j = i + 1
                    while j < len(s) and '0' <= s[j] <= '7' and (j - i - 1) < 3: j += 1
                    try:
                        octal_val = int(s[i + 1:j], 8)
                        out += chr(octal_val)
                    except ValueError:
                        logging.warning(f"Invalid octal escape sequence: \\{s[i + 1:j]}")
                        out += s[
                            i]  # Keep backslash if invalid
                    i = j
                    continue
                elif s[i + 1] in ('x', 'X'):  # 十六进制
                    j = i + 2
                    while j < len(s) and s[j] in '0123456789abcdefABCDEF': j += 1
                    if j > i + 2:
                        try:
                            hex_val = int(s[i + 2:j], 16)
                            out += chr(hex_val)
                        except ValueError:
                            logging.warning(f"Invalid hex escape sequence: \\{s[i + 1:j]}")
                            out += s[i]
                        i = j
                        continue
                    else:  # \x 后没有有效数字
                        logging.warning(f"Incomplete hex escape sequence: \\{s[i + 1:]}")
                        out += s[i]
                        i += 1
                        continue
                else:  # 其他未识别的转义符 (例如 \c)，通常行为是忽略反斜杠
                    out += s[i + 1]
                    i += 2
                    continue
            else:  # 字符串末尾的反斜杠
                out += s[i]
                i += 1
                continue
        out += s[i]
        i += 1
    return out


class Lexer:
    def __init__(self, code, line_mapping):
        if code is None:
            raise ValueError("Input code cannot be None")
        self.code = code
        self.position = 0
        self.processed_line = 1
        self.column = 1
        self.tokens = []
        if not isinstance(line_mapping, dict):
            logging.warning("Invalid line_mapping provided to Lexer, creating default map.")
            num_lines = code.count('\n') + 1
            self.line_mapping = {i: i for i in range(1, num_lines + 1)}
        else:
            self.line_mapping = line_mapping

        self.keywords = {
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool',
            'break', 'case', 'catch', 'char', 'class', 'compl', 'concept', 'const',
            'const_cast', 'consteval', 'constinit', 'constexpr', 'continue', 'decltype',
            'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
            'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline',
            'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq',
            'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public',
            'reinterpret_cast', 'register', 'requires', 'restrict', 'return', 'short',
            'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct',
            'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typeid',
            'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
            'wchar_t', 'while', 'xor', 'xor_eq', 'string'
        }
        self.token_specs = [
            ('COMMENT_MULTI', r'/\*.*?\*/'),  # 多行注释
            ('COMMENT_SINGLE', r'//.*'),  # 单行注释
            ('NEWLINE', r'\n'),  # 换行符
            ('WHITESPACE', r'[ \t\r\f\v]+'),  # 空白符
            # 原始字符串 R"(...)"
            ('RAW_STRING_LITERAL', r'R"((?P<delim>[^()\s]{0,16}))\((?P<content>.*?)\)\1"'),
            # 字符字面量 'c', L'c', u'c', U'c', u8'c'
            ('CHAR_LITERAL', r'(u8|[uUL])?\'([^\\\'\n]|\\([nrtv\\\'\"?abf0]|x[0-9a-fA-F]+|[0-7]{1,3}))+\''),
            # 允许多个字符或转义序列
            # 字符串字面量 "...", L"...", u"...", U"...", u8"..."
            ('STRING_LITERAL', r'(u8|[uUL])?\"([^\\\"\n]|\\([nrtv\\\'\"?abf0]|x[0-9a-fA-F]+|[0-7]{1,3}))*\"'),
            # 运算符 (顺序很重要，长的优先)
            ('OPERATOR_3CHAR', r'<<=|>>='),  # 三字符运算符
            ('OPERATOR_ELLIPSIS', r'\.\.\.'),  # 省略号
            ('OPERATOR_2CHAR', r'->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|::'),  # 双字符运算符
            # --- MODIFIED: 逗号 ',' 移动到 PUNCTUATOR ---
            ('OPERATOR_1CHAR', r'[+\-*/%&|^~!<>=?.:]'),  # 单字符运算符 (移除逗号)
            # --- MODIFIED: 添加逗号 ',' ---
            ('PUNCTUATOR', r'[{}();\[\],]'),  # 标点符号 (添加逗号)
            # 数字字面量 (允许数字分隔符 ')
            ('BIN_INTEGER', r'0[bB][01]+(?:[01\']*[01])?[uUlL]*'),  # 二进制 (修正分隔符处理)
            ('HEX_INTEGER', r'0[xX][0-9a-fA-F]+(?:[0-9a-fA-F\']*[0-9a-fA-F])?[uUlL]*'),  # 十六进制 (修正分隔符处理)
            # 浮点数 (改进以处理各种形式)
            ('FLOAT_LITERAL',
             r'(?:(?:[0-9]+(?:\'[0-9]+)*\.(?:[0-9](?:\'[0-9]+)*)*|\.[0-9]+(?:\'[0-9]+)*)(?:[eE][+-]?[0-9]+(?:\'[0-9]+)*)?|[0-9]+(?:\'[0-9]+)*[eE][+-]?[0-9]+(?:\'[0-9]+)*)[fFlL]?'),
            ('OCT_INTEGER', r'0[0-7]+(?:[0-7\']*[0-7])?[uUlL]*'),  # 八进制 (修正分隔符处理)
            ('DEC_INTEGER', r'[1-9][0-9]*(?:[0-9\']*[0-9])?[uUlL]*|0[uUlL]*'),  # 十进制 (修正分隔符处理, 包含单独的0)
            # 标识符
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            # 不匹配任何规则的字符
            ('MISMATCH', r'.')
        ]
        self.master_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.token_specs),
                                       re.DOTALL)

    def _get_original_line(self, processed_line_num):
        """根据处理后的行号获取原始行号。"""
        # 查找最接近的映射行号
        # 如果 line_mapping 为空或无效，返回处理行号
        if not self.line_mapping: return processed_line_num
        # 找到第一个大于等于当前处理行号的映射键
        target_line = processed_line_num
        for proc_line in sorted(self.line_mapping.keys()):
            if proc_line >= processed_line_num:
                target_line = self.line_mapping[proc_line]
                break
        else:  # 如果循环完成没有找到更大的，取最后一个映射值
            if self.line_mapping:
                target_line = self.line_mapping[max(self.line_mapping.keys())]

        return target_line if target_line is not None else processed_line_num

    def tokenize(self):
        """执行词法分析，返回 Token 列表。"""
        while self.position < len(self.code):
            match = self.master_regex.match(self.code, self.position)
            if not match:
                # 无法匹配，报告错误
                original_error_line = self._get_original_line(self.processed_line)
                raise LexerError(f"Unable to match token near '{self.code[self.position:self.position + 10]}...'",
                                 original_error_line, self.column)

            kind = match.lastgroup
            value = match.group(kind)
            start_column = self.column
            current_original_line = self._get_original_line(self.processed_line)

            # --- 跳过空白和注释 ---
            if kind in ['COMMENT_MULTI', 'COMMENT_SINGLE', 'NEWLINE', 'WHITESPACE']:
                lines_in_match = value.count('\n')
                if lines_in_match > 0:
                    self.processed_line += lines_in_match
                    # 列号重置为换行符后的字符数
                    self.column = len(value) - value.rfind('\n')
                else:
                    # 没有换行，仅增加列号
                    self.column += len(value)
                self.position = match.end()  # 更新位置
                continue  # 继续下一轮匹配

            # --- 处理有效 Token ---
            token_type = None
            token_value = value  # 默认值

            # 根据匹配的组名确定 Token 类型
            if kind in ['OPERATOR_3CHAR', 'OPERATOR_2CHAR', 'OPERATOR_1CHAR', 'OPERATOR_ELLIPSIS']:
                token_type = 'OPERATOR'
            elif kind == 'PUNCTUATOR':
                token_type = 'PUNCTUATOR'
            elif kind == 'CHAR_LITERAL':
                token_type = 'CHAR_LITERAL'
                # 提取引号内的内容并处理转义
                prefix_match = re.match(r'(u8|[uUL])?', token_value)
                prefix_len = prefix_match.end() if prefix_match else 0
                # +1 跳过前缀和开头的单引号, -1 跳过结尾的单引号
                char_content = token_value[prefix_len + 1: -1]
                token_value = _interpret_escapes(char_content)
                # C++ 字符字面量通常只包含一个字符（或转义表示的一个字符）
                if len(token_value) != 1:
                    logging.warning(f"Multi-character char literal L{current_original_line}:C{start_column}: {value}")
            elif kind == 'STRING_LITERAL':
                token_type = 'STRING_LITERAL'
                prefix_match = re.match(r'(u8|[uUL])?', token_value)
                prefix_len = prefix_match.end() if prefix_match else 0
                # +1 跳过前缀和开头的双引号, -1 跳过结尾的双引号
                string_content = token_value[prefix_len + 1: -1]
                token_value = _interpret_escapes(string_content)
            elif kind == 'RAW_STRING_LITERAL':
                token_type = 'STRING_LITERAL'  # 类型仍是字符串
                token_value = match.group('content')  # 值是括号内的内容，无需转义
            elif kind in ['HEX_INTEGER', 'OCT_INTEGER', 'DEC_INTEGER', 'BIN_INTEGER']:
                token_type = 'INTEGER_LITERAL'
                # 保留原始字符串值，解析器或后续阶段进行数值转换
                token_value = value.replace("'", "")  # 去除数字分隔符
            elif kind == 'FLOAT_LITERAL':
                token_type = 'FLOAT_LITERAL'
                token_value = value.replace("'", "")  # 去除数字分隔符
            elif kind == 'IDENTIFIER':
                # 检查是否是关键字
                token_type = 'KEYWORD' if token_value in self.keywords else 'IDENTIFIER'
            elif kind == 'MISMATCH':
                # 遇到无法识别的字符，抛出错误
                raise LexerError(f"Illegal character encountered: '{token_value}'", current_original_line, start_column)
            else:
                # 不应到达这里，所有模式都应被处理或标记为 MISMATCH
                raise LexerError(f"Internal Lexer Error: Unhandled token kind: {kind}", current_original_line,
                                 start_column)

            # 创建 Token 对象并添加到列表
            self.tokens.append(Token(token_type, token_value, current_original_line, start_column))

            # 更新位置和行列号
            # 注意：使用 match.group(kind) 的长度来更新，因为 token_value 可能已被修改（如处理转义）
            matched_text_length = len(match.group(kind))
            lines_in_value = match.group(kind).count('\n')  # 不应有换行，除非正则错误
            if lines_in_value > 0:
                # 这通常不应发生在非空白/注释 token 中，但以防万一
                self.processed_line += lines_in_value
                self.column = matched_text_length - match.group(kind).rfind('\n')
            else:
                self.column += matched_text_length

            self.position = match.end()  # 更新扫描位置

        # 添加文件结束符 (EOF) Token
        eof_original_line = self._get_original_line(self.processed_line)
        self.tokens.append(Token("EOF", "", eof_original_line, self.column))
        return self.tokens


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
        # --- 可选的预处理步骤 ---
        if BasicPreprocessor:  # 检查预处理器是否成功导入
            try:
                logging.info("正在尝试运行预处理器...")
                preprocessor = BasicPreprocessor()
                processed_code, line_map = preprocessor.process(raw_code)
                logging.info("预处理完成.")
            except Exception as e:
                logging.error(f"预处理时发生错误: {e}。将对原始代码进行词法分析。", exc_info=True)
                processed_code = raw_code  # 出错则回退到原始代码
                num_lines = raw_code.count('\n') + 1
                line_map = {i: i for i in range(1, num_lines + 1)}  # 基础行号映射
        else:  # 没有预处理器
            logging.warning("未找到预处理器，将直接对原始代码进行词法分析。")
            processed_code = raw_code
            num_lines = raw_code.count('\n') + 1
            line_map = {i: i for i in range(1, num_lines + 1)}
        # --- 执行词法分析 ---
        if processed_code is not None:
            logging.info("正在进行词法分析...")
            lexer = Lexer(processed_code, line_map)
            tokens = lexer.tokenize()
            logging.info("词法分析完成.")
            print("\n词法分析结果 (Tokens):")
            for token in tokens:
                print(token)  # 打印每个 token
        else:
            logging.error("错误：没有可供词法分析的代码。")
            sys.exit(1)
    except FileNotFoundError:
        logging.error(f"错误: 文件 '{input_file_path}' 未找到")
        sys.exit(1)
    except LexerError as e:
        logging.error(e)  # 打印词法错误
        sys.exit(1)
    except Exception as e:  # 捕获其他意外错误
        logging.error(f"发生未预料的错误: {e}", exc_info=True)
        sys.exit(1)
