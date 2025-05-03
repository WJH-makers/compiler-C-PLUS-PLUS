# coding=utf-8
import unittest
import io
import sys

try:
    from lexer import Lexer, Token
    from parser import Parser, ParseError
    from compiler_ast import (
        Program, FunctionDefinition, DeclarationStatement, ExpressionStatement,
        Identifier, IntegerLiteral, FloatLiteral, StringLiteral, CharLiteral,
        BinaryOp, UnaryOp, CallExpression, IfStatement, ForStatement,
        WhileStatement, DoWhileStatement, BreakStatement, ContinueStatement,
        ReturnStatement, CompoundStatement, Parameter, ASTNode
    )
except ImportError as e:
    print(f"Import Error: Make sure _lexer.py, _parser.py, _ast.py are accessible.")
    print(e)
    sys.exit(1)


# Helper function to capture print output (for error messages) if needed
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class TestParser(unittest.TestCase):

    def _parse_code(self, code_string):
        """Helper method to lex and parse code. Returns AST root or raises ParseError."""
        # print(f"\n--- Testing Code ---\n{code_string}\n--------------------") # Uncomment for debug
        lexer = Lexer(code_string)
        tokens = lexer.tokenize()
        # print("Tokens:", tokens) # Uncomment for debug
        parser = Parser(tokens)
        ast = parser.parse_program()
        # print("AST:", ast) # Uncomment for debug
        return ast

    def assertASTNode(self, node, expected_type, msg=None):
        """Asserts that a node is an instance of a specific AST type."""
        self.assertIsInstance(node, expected_type, msg or f"Expected node type {expected_type}, got {type(node)}")

    # --- Tests for Valid Code ---

    def test_empty_program(self):
        code = ""
        ast = self._parse_code(code)
        self.assertASTNode(ast, Program)
        self.assertEqual(len(ast.declarations), 0, "Empty program should have 0 declarations")

    def test_simple_function(self):
        code = """
        int main() {
            return 0;
        }
        """
        ast = self._parse_code(code)
        self.assertEqual(len(ast.declarations), 1)
        func_def = ast.declarations[0]
        self.assertASTNode(func_def, FunctionDefinition)
        self.assertEqual(func_def.return_type, "int")
        self.assertASTNode(func_def.name, Identifier)
        self.assertEqual(func_def.name.name, "main")
        self.assertEqual(len(func_def.params), 0)
        self.assertASTNode(func_def.body, CompoundStatement)
        self.assertEqual(len(func_def.body.statements), 1)
        ret_stmt = func_def.body.statements[0]
        self.assertASTNode(ret_stmt, ReturnStatement)
        self.assertASTNode(ret_stmt.value, IntegerLiteral)
        self.assertEqual(ret_stmt.value.value, 0)

    def test_global_variable(self):
        code = "float pi = 3.14;"
        ast = self._parse_code(code)
        self.assertEqual(len(ast.declarations), 1)
        decl = ast.declarations[0]
        self.assertASTNode(decl, DeclarationStatement)
        self.assertEqual(decl.decl_type, "float")
        self.assertASTNode(decl.name, Identifier)
        self.assertEqual(decl.name.name, "pi")
        self.assertASTNode(decl.initializer, FloatLiteral)
        self.assertAlmostEqual(decl.initializer.value, 3.14)  # Use assertAlmostEqual for floats

    def test_declaration_and_assignment_expression(self):
        # Testing x = 5 + 3; where assignment is parsed as BinaryOp '='
        code = """
        void test() {
            int x;
            x = 5 + 3;
        }
        """
        ast = self._parse_code(code)
        func_def = ast.declarations[0]
        body = func_def.body
        self.assertEqual(len(body.statements), 2)
        decl = body.statements[0]
        assign_stmt = body.statements[1]  # This is an ExpressionStatement

        self.assertASTNode(decl, DeclarationStatement)
        self.assertEqual(decl.name.name, "x")
        self.assertIsNone(decl.initializer)

        self.assertASTNode(assign_stmt, ExpressionStatement)
        assign_expr = assign_stmt.expression  # This is the BinaryOp '='
        self.assertASTNode(assign_expr, BinaryOp)
        self.assertEqual(assign_expr.op, '=')
        self.assertASTNode(assign_expr.left, Identifier)
        self.assertEqual(assign_expr.left.name, 'x')
        # Right side is the BinaryOp '+'
        add_expr = assign_expr.right
        self.assertASTNode(add_expr, BinaryOp)
        self.assertEqual(add_expr.op, '+')
        self.assertASTNode(add_expr.left, IntegerLiteral)
        self.assertEqual(add_expr.left.value, 5)
        self.assertASTNode(add_expr.right, IntegerLiteral)
        self.assertEqual(add_expr.right.value, 3)

    def test_precedence_and_associativity(self):
        code = "int main() { return 2 + 3 * 4 - 8 / 2; }"  # Expected AST for: ((2 + (3 * 4)) - (8 / 2)) = 11
        ast = self._parse_code(code)
        ret_stmt = ast.declarations[0].body.statements[0]
        expr = ret_stmt.value
        # Top level op should be '-' due to left associativity of +/-
        self.assertASTNode(expr, BinaryOp)
        self.assertEqual(expr.op, '-')

        # Left side of '-' is (2 + (3*4))
        left_minus = expr.left
        self.assertASTNode(left_minus, BinaryOp)
        self.assertEqual(left_minus.op, '+')
        self.assertASTNode(left_minus.left, IntegerLiteral)  # 2
        self.assertEqual(left_minus.left.value, 2)
        mult_expr = left_minus.right  # 3 * 4
        self.assertASTNode(mult_expr, BinaryOp)
        self.assertEqual(mult_expr.op, '*')
        self.assertASTNode(mult_expr.left, IntegerLiteral)  # 3
        self.assertEqual(mult_expr.left.value, 3)
        self.assertASTNode(mult_expr.right, IntegerLiteral)  # 4
        self.assertEqual(mult_expr.right.value, 4)

        # Right side of '-' is (8 / 2)
        right_minus = expr.right
        self.assertASTNode(right_minus, BinaryOp)
        self.assertEqual(right_minus.op, '/')
        self.assertASTNode(right_minus.left, IntegerLiteral)  # 8
        self.assertEqual(right_minus.left.value, 8)
        self.assertASTNode(right_minus.right, IntegerLiteral)  # 2
        self.assertEqual(right_minus.right.value, 2)

    def test_parentheses_override_precedence(self):
        code = "int main() { return (2 + 3) * 4; }"  # Expected AST for: ((2 + 3) * 4) = 20
        ast = self._parse_code(code)
        ret_stmt = ast.declarations[0].body.statements[0]
        expr = ret_stmt.value
        # Top level op should be '*'
        self.assertASTNode(expr, BinaryOp)
        self.assertEqual(expr.op, '*')
        # Left side is (2 + 3)
        add_expr = expr.left
        self.assertASTNode(add_expr, BinaryOp)
        self.assertEqual(add_expr.op, '+')
        self.assertASTNode(add_expr.left, IntegerLiteral)  # 2
        self.assertEqual(add_expr.left.value, 2)
        self.assertASTNode(add_expr.right, IntegerLiteral)  # 3
        self.assertEqual(add_expr.right.value, 3)
        # Right side is 4
        self.assertASTNode(expr.right, IntegerLiteral)  # 4
        self.assertEqual(expr.right.value, 4)

    def test_unary_operators(self):
        code = "int main() { return -5 + +3 - !0; }"  # Expected: ((-5) + (+3)) - (!0) = (-2) - 1 = -3
        ast = self._parse_code(code)
        expr = ast.declarations[0].body.statements[0].value
        # Top level op should be '-'
        self.assertASTNode(expr, BinaryOp)
        self.assertEqual(expr.op, '-')
        # Left side of outer '-' is ((-5) + (+3))
        left_outer_minus = expr.left
        self.assertASTNode(left_outer_minus, BinaryOp)
        self.assertEqual(left_outer_minus.op, '+')
        unary_minus = left_outer_minus.left
        self.assertASTNode(unary_minus, UnaryOp)
        self.assertEqual(unary_minus.op, '-')
        self.assertASTNode(unary_minus.operand, IntegerLiteral)  # 5
        self.assertEqual(unary_minus.operand.value, 5)
        unary_plus = left_outer_minus.right
        self.assertASTNode(unary_plus, UnaryOp)
        self.assertEqual(unary_plus.op, '+')
        self.assertASTNode(unary_plus.operand, IntegerLiteral)  # 3
        self.assertEqual(unary_plus.operand.value, 3)
        # Right side of outer '-' is (!0)
        unary_not = expr.right
        self.assertASTNode(unary_not, UnaryOp)
        self.assertEqual(unary_not.op, '!')
        self.assertASTNode(unary_not.operand, IntegerLiteral)  # 0
        self.assertEqual(unary_not.operand.value, 0)

    def test_function_call_expression(self):
        code = """
        int calculate(int a, char* msg) { return a; }
        int main() {
            int result;
            result = calculate(5 * 2, "hello") + 1; // Call inside expression
        }
        """
        ast = self._parse_code(code)
        main_func = ast.declarations[1]  # main is the second function
        assign_stmt = main_func.body.statements[1]  # ExpressionStatement
        self.assertASTNode(assign_stmt, ExpressionStatement)
        assign_expr = assign_stmt.expression  # BinaryOp '='
        self.assertASTNode(assign_expr, BinaryOp)
        self.assertEqual(assign_expr.op, '=')
        self.assertEqual(assign_expr.left.name, 'result')

        add_expr = assign_expr.right  # calculate(...) + 1
        self.assertASTNode(add_expr, BinaryOp)
        self.assertEqual(add_expr.op, '+')
        self.assertASTNode(add_expr.right, IntegerLiteral)  # 1
        self.assertEqual(add_expr.right.value, 1)

        call_expr = add_expr.left  # calculate(...)
        self.assertASTNode(call_expr, CallExpression)
        self.assertASTNode(call_expr.function, Identifier)
        self.assertEqual(call_expr.function.name, "calculate")
        self.assertEqual(len(call_expr.args), 2)
        # First arg: 5 * 2
        arg1 = call_expr.args[0]
        self.assertASTNode(arg1, BinaryOp)
        self.assertEqual(arg1.op, '*')
        self.assertEqual(arg1.left.value, 5)
        self.assertEqual(arg1.right.value, 2)
        # Second arg: "hello"
        arg2 = call_expr.args[1]
        self.assertASTNode(arg2, StringLiteral)
        self.assertEqual(arg2.value, "hello")

    def test_for_loop_all_clauses(self):
        code = """
        void loop() {
            int sum = 0;
            int i;
            for (i = 1; i <= 10; i = i + 1) {
                sum = sum + i;
            }
        }
        """
        ast = self._parse_code(code)
        func_def = ast.declarations[0]
        for_stmt = func_def.body.statements[2]  # 0: sum=0, 1: int i, 2: for loop
        self.assertASTNode(for_stmt, ForStatement)
        # Check init: i = 1 (ExpressionStatement(BinaryOp('=')))
        self.assertASTNode(for_stmt.init, ExpressionStatement)
        self.assertASTNode(for_stmt.init.expression, BinaryOp)
        self.assertEqual(for_stmt.init.expression.op, '=')
        self.assertEqual(for_stmt.init.expression.left.name, 'i')
        self.assertEqual(for_stmt.init.expression.right.value, 1)
        # Check condition: i <= 10 (BinaryOp('<='))
        self.assertASTNode(for_stmt.condition, BinaryOp)
        self.assertEqual(for_stmt.condition.op, '<=')
        self.assertEqual(for_stmt.condition.left.name, 'i')
        self.assertEqual(for_stmt.condition.right.value, 10)
        # Check update: i = i + 1 (BinaryOp('='))
        self.assertASTNode(for_stmt.update, BinaryOp)
        self.assertEqual(for_stmt.update.op, '=')
        self.assertEqual(for_stmt.update.left.name, 'i')
        self.assertASTNode(for_stmt.update.right, BinaryOp)  # i + 1
        self.assertEqual(for_stmt.update.right.op, '+')
        # Check body
        self.assertASTNode(for_stmt.body, CompoundStatement)
        self.assertEqual(len(for_stmt.body.statements), 1)

    def test_for_loop_declaration_init(self):
        code = """
        void loop() {
             for (int i = 0; i < 5;) { /* Empty update */ }
        }
        """
        ast = self._parse_code(code)
        for_stmt = ast.declarations[0].body.statements[0]
        self.assertASTNode(for_stmt, ForStatement)
        # Check init: int i = 0 (DeclarationStatement)
        self.assertASTNode(for_stmt.init, DeclarationStatement)
        self.assertEqual(for_stmt.init.decl_type, 'int')
        self.assertEqual(for_stmt.init.name.name, 'i')
        self.assertEqual(for_stmt.init.initializer.value, 0)
        # Check condition: i < 5
        self.assertASTNode(for_stmt.condition, BinaryOp)
        self.assertEqual(for_stmt.condition.op, '<')
        # Check update: None
        self.assertIsNone(for_stmt.update)
        # Check body
        self.assertASTNode(for_stmt.body, CompoundStatement)

    def test_do_while_loop(self):
        code = """
         void doloop() {
             int x = 10;
             do {
                 x = x - 1; // Statement
             } while (x > 0);
         }
         """
        ast = self._parse_code(code)
        func_def = ast.declarations[0]
        do_while_stmt = func_def.body.statements[1]
        self.assertASTNode(do_while_stmt, DoWhileStatement)
        # Check body
        self.assertASTNode(do_while_stmt.body, CompoundStatement)
        self.assertEqual(len(do_while_stmt.body.statements), 1)
        assign_stmt = do_while_stmt.body.statements[0]
        self.assertASTNode(assign_stmt, ExpressionStatement)
        # Check condition: x > 0
        self.assertASTNode(do_while_stmt.condition, BinaryOp)
        self.assertEqual(do_while_stmt.condition.op, '>')
        self.assertEqual(do_while_stmt.condition.left.name, 'x')
        self.assertEqual(do_while_stmt.condition.right.value, 0)

    def test_break_continue(self):
        code = """
         void control() {
             int i = 0;
             while(i < 10) {
                 i = i + 1;
                 if (i == 5) { continue; }
                 if (i == 8) { break; }
                 work(i); // Assume exists
             }
         }
         """
        ast = self._parse_code(code)
        while_stmt = ast.declarations[0].body.statements[1]
        inner_block = while_stmt.body  # Compound statement
        self.assertEqual(len(inner_block.statements), 4)  # assign, if-continue, if-break, call
        if_continue = inner_block.statements[1]
        if_break = inner_block.statements[2]
        # Check inside the 'if' bodies
        self.assertASTNode(if_continue.then_branch, CompoundStatement)
        self.assertASTNode(if_continue.then_branch.statements[0], ContinueStatement)
        self.assertASTNode(if_break.then_branch, CompoundStatement)
        self.assertASTNode(if_break.then_branch.statements[0], BreakStatement)

    def test_char_literal(self):
        code = "char nl = '\\n';"
        ast = self._parse_code(code)
        decl = ast.declarations[0]
        self.assertASTNode(decl.initializer, CharLiteral)
        self.assertEqual(decl.initializer.value, '\n')  # Check interpreted value

    def test_pointer_type(self):
        code = "int* ptr;"
        ast = self._parse_code(code)
        decl = ast.declarations[0]
        self.assertEqual(decl.decl_type, "int*")

    # --- Tests for Invalid Code ---

    def test_error_missing_semicolon_statement(self):
        code = """
        int main() {
            int x = 1
            return 0;
        }
        """
        # Error should be caught after '1'
        with self.assertRaisesRegex(ParseError, r"Expected ';' after declaration"):
            self._parse_code(code)

    def test_error_missing_semicolon_return(self):
        code = """
        int main() {
            return 0 // Missing semicolon
        }
        """
        with self.assertRaisesRegex(ParseError, r"Expected ';' after return statement"):
            self._parse_code(code)

    def test_error_mismatched_parenthesis_expr(self):
        code = "int main() { int y = (5 + 3; }"  # Mismatched paren in expression
        with self.assertRaisesRegex(ParseError, r"Expected '\)' after parenthesized expression"):
            self._parse_code(code)

    def test_error_mismatched_parenthesis_if(self):
        code = "int main() { if (x > 0 { return 1; } }"  # Mismatched paren after if condition
        with self.assertRaisesRegex(ParseError, r"Expected '\)' after 'if' condition"):
            self._parse_code(code)

    def test_error_mismatched_braces(self):
        code = "int main() { return 0;"  # Missing closing brace
        with self.assertRaisesRegex(ParseError, r"Expected '\}' to end block, found EOF"):
            self._parse_code(code)

    def test_error_unexpected_token_in_expression(self):
        code = "int main() { int x = 5 + < ; }"  # Unexpected token '<'
        # Error message might depend on context, assertRaises is safer
        with self.assertRaises(ParseError):
            self._parse_code(code)
        # Example using regex if message is predictable:
        # with self.assertRaisesRegex(ParseError, r"Unexpected token in expression: <"):
        #      self._parse_code(code)

    def test_error_invalid_assignment_lhs(self):
        code = "int main() { 5 = 10; }"  # Invalid LHS for assignment
        # The check for this specific error (LHS must be lvalue) was added in parser
        with self.assertRaisesRegex(ParseError, r"Invalid left-hand side in assignment"):
            self._parse_code(code)

    def test_error_missing_condition_paren_while(self):
        code = "int main() { while x < 5) {} }"  # Missing opening paren
        with self.assertRaisesRegex(ParseError, r"Expected '\(' after 'while'"):
            self._parse_code(code)

    def test_error_missing_condition_paren_for(self):
        code = "int main() { for int i=0; i<10; i=i+1) {} }"  # Missing opening paren
        with self.assertRaisesRegex(ParseError, r"Expected '\(' after 'for'"):
            self._parse_code(code)

    def test_error_incomplete_for_loop(self):
        code = "int main() { for (i=0; i<10 ) {} }"  # Missing semicolon and closing paren
        with self.assertRaisesRegex(ParseError, r"Expected ';' after 'for' condition expression"):
            self._parse_code(code)


if __name__ == '__main__':
    print("Running Parser Tests...")
    unittest.main(verbosity=2)  # verbosity=2 provides more detailed output
