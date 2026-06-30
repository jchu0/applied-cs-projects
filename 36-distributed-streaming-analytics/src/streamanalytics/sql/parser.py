"""SQL Parser - Recursive descent parser for SQL statements.

Supports SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT,
JOIN, subqueries, UNION/INTERSECT/EXCEPT, and common expressions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .ast import (
    Expression, Literal, ColumnRef, BinaryOp, UnaryOp, FunctionCall,
    AggregateExpr, CaseExpr, SubqueryExpr, InList, Between, Like, IsNull, Cast,
    SelectStatement, FromClause, JoinClause, WhereClause, GroupByClause,
    HavingClause, OrderByClause, OrderByItem, LimitClause, TableRef,
    DataType, JoinType, AggregateFunction, SortOrder, NullOrdering,
)


class TokenType(Enum):
    """SQL token types."""

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()
    QUOTED_IDENTIFIER = auto()

    # Keywords
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP = auto()
    BY = auto()
    HAVING = auto()
    ORDER = auto()
    LIMIT = auto()
    OFFSET = auto()
    DISTINCT = auto()
    ALL = auto()
    AS = auto()

    # Joins
    JOIN = auto()
    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()
    OUTER = auto()
    CROSS = auto()
    ON = auto()
    USING = auto()

    # Set operations
    UNION = auto()
    INTERSECT = auto()
    EXCEPT = auto()

    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    BETWEEN = auto()
    LIKE = auto()
    IS = auto()
    NULL = auto()
    TRUE = auto()
    FALSE = auto()
    EXISTS = auto()

    # Aggregate functions
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()

    # Other keywords
    CASE = auto()
    WHEN = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    CAST = auto()
    OVER = auto()
    PARTITION = auto()
    ROWS = auto()
    RANGE = auto()
    FILTER = auto()
    NULLS = auto()
    FIRST = auto()
    LAST = auto()
    ASC = auto()
    DESC = auto()
    WITH = auto()
    ROLLUP = auto()
    CUBE = auto()
    GROUPING = auto()
    SETS = auto()
    ESCAPE = auto()
    ANY = auto()
    SOME = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    CONCAT = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    DOT = auto()
    SEMICOLON = auto()
    COLON = auto()
    DOUBLE_COLON = auto()

    # Special
    EOF = auto()
    ERROR = auto()


# Keywords mapping
KEYWORDS = {
    "select": TokenType.SELECT,
    "from": TokenType.FROM,
    "where": TokenType.WHERE,
    "group": TokenType.GROUP,
    "by": TokenType.BY,
    "having": TokenType.HAVING,
    "order": TokenType.ORDER,
    "limit": TokenType.LIMIT,
    "offset": TokenType.OFFSET,
    "distinct": TokenType.DISTINCT,
    "all": TokenType.ALL,
    "as": TokenType.AS,
    "join": TokenType.JOIN,
    "inner": TokenType.INNER,
    "left": TokenType.LEFT,
    "right": TokenType.RIGHT,
    "full": TokenType.FULL,
    "outer": TokenType.OUTER,
    "cross": TokenType.CROSS,
    "on": TokenType.ON,
    "using": TokenType.USING,
    "union": TokenType.UNION,
    "intersect": TokenType.INTERSECT,
    "except": TokenType.EXCEPT,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "in": TokenType.IN,
    "between": TokenType.BETWEEN,
    "like": TokenType.LIKE,
    "is": TokenType.IS,
    "null": TokenType.NULL,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "exists": TokenType.EXISTS,
    "count": TokenType.COUNT,
    "sum": TokenType.SUM,
    "avg": TokenType.AVG,
    "min": TokenType.MIN,
    "max": TokenType.MAX,
    "case": TokenType.CASE,
    "when": TokenType.WHEN,
    "then": TokenType.THEN,
    "else": TokenType.ELSE,
    "end": TokenType.END,
    "cast": TokenType.CAST,
    "over": TokenType.OVER,
    "partition": TokenType.PARTITION,
    "rows": TokenType.ROWS,
    "range": TokenType.RANGE,
    "filter": TokenType.FILTER,
    "nulls": TokenType.NULLS,
    "first": TokenType.FIRST,
    "last": TokenType.LAST,
    "asc": TokenType.ASC,
    "desc": TokenType.DESC,
    "with": TokenType.WITH,
    "rollup": TokenType.ROLLUP,
    "cube": TokenType.CUBE,
    "grouping": TokenType.GROUPING,
    "sets": TokenType.SETS,
    "escape": TokenType.ESCAPE,
    "any": TokenType.ANY,
    "some": TokenType.SOME,
}


@dataclass
class Token:
    """SQL token."""

    type: TokenType
    value: str
    position: int = 0
    line: int = 1
    column: int = 1


class SQLLexer:
    """SQL tokenizer/lexer."""

    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        self._tokenize()

    def _tokenize(self) -> None:
        """Tokenize the entire SQL string."""
        while self.pos < len(self.sql):
            self._skip_whitespace()
            if self.pos >= len(self.sql):
                break

            token = self._next_token()
            if token.type != TokenType.ERROR or token.value:
                self.tokens.append(token)

        self.tokens.append(Token(TokenType.EOF, "", self.pos, self.line, self.column))

    def _skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while self.pos < len(self.sql):
            c = self.sql[self.pos]
            if c in " \t\r":
                self.pos += 1
                self.column += 1
            elif c == "\n":
                self.pos += 1
                self.line += 1
                self.column = 1
            elif c == "-" and self.pos + 1 < len(self.sql) and self.sql[self.pos + 1] == "-":
                # Line comment
                while self.pos < len(self.sql) and self.sql[self.pos] != "\n":
                    self.pos += 1
            elif c == "/" and self.pos + 1 < len(self.sql) and self.sql[self.pos + 1] == "*":
                # Block comment
                self.pos += 2
                while self.pos + 1 < len(self.sql):
                    if self.sql[self.pos] == "*" and self.sql[self.pos + 1] == "/":
                        self.pos += 2
                        break
                    if self.sql[self.pos] == "\n":
                        self.line += 1
                        self.column = 1
                    self.pos += 1
            else:
                break

    def _next_token(self) -> Token:
        """Get the next token."""
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        c = self.sql[self.pos]

        # String literal
        if c in "'\"":
            return self._string_literal(c, start_pos, start_line, start_col)

        # Quoted identifier
        if c == '"' or c == '`':
            return self._quoted_identifier(c, start_pos, start_line, start_col)

        # Number
        if c.isdigit() or (c == "." and self.pos + 1 < len(self.sql) and self.sql[self.pos + 1].isdigit()):
            return self._number(start_pos, start_line, start_col)

        # Identifier or keyword
        if c.isalpha() or c == "_":
            return self._identifier(start_pos, start_line, start_col)

        # Operators and punctuation
        return self._operator(start_pos, start_line, start_col)

    def _string_literal(self, quote: str, start_pos: int, start_line: int, start_col: int) -> Token:
        """Parse a string literal."""
        self.pos += 1
        self.column += 1
        value = []

        while self.pos < len(self.sql):
            c = self.sql[self.pos]
            if c == quote:
                # Check for escaped quote
                if self.pos + 1 < len(self.sql) and self.sql[self.pos + 1] == quote:
                    value.append(quote)
                    self.pos += 2
                    self.column += 2
                else:
                    self.pos += 1
                    self.column += 1
                    return Token(TokenType.STRING, "".join(value), start_pos, start_line, start_col)
            else:
                value.append(c)
                self.pos += 1
                if c == "\n":
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1

        return Token(TokenType.ERROR, "Unterminated string", start_pos, start_line, start_col)

    def _quoted_identifier(self, quote: str, start_pos: int, start_line: int, start_col: int) -> Token:
        """Parse a quoted identifier."""
        self.pos += 1
        self.column += 1
        value = []

        closing_quote = '"' if quote == '"' else '`'
        while self.pos < len(self.sql):
            c = self.sql[self.pos]
            if c == closing_quote:
                self.pos += 1
                self.column += 1
                return Token(TokenType.QUOTED_IDENTIFIER, "".join(value), start_pos, start_line, start_col)
            value.append(c)
            self.pos += 1
            self.column += 1

        return Token(TokenType.ERROR, "Unterminated identifier", start_pos, start_line, start_col)

    def _number(self, start_pos: int, start_line: int, start_col: int) -> Token:
        """Parse a numeric literal."""
        value = []
        has_dot = False
        has_exp = False

        while self.pos < len(self.sql):
            c = self.sql[self.pos]
            if c.isdigit():
                value.append(c)
                self.pos += 1
                self.column += 1
            elif c == "." and not has_dot and not has_exp:
                value.append(c)
                has_dot = True
                self.pos += 1
                self.column += 1
            elif c in "eE" and not has_exp:
                value.append(c)
                has_exp = True
                self.pos += 1
                self.column += 1
                if self.pos < len(self.sql) and self.sql[self.pos] in "+-":
                    value.append(self.sql[self.pos])
                    self.pos += 1
                    self.column += 1
            else:
                break

        value_str = "".join(value)
        token_type = TokenType.FLOAT if has_dot or has_exp else TokenType.INTEGER
        return Token(token_type, value_str, start_pos, start_line, start_col)

    def _identifier(self, start_pos: int, start_line: int, start_col: int) -> Token:
        """Parse an identifier or keyword."""
        value = []
        while self.pos < len(self.sql):
            c = self.sql[self.pos]
            if c.isalnum() or c == "_":
                value.append(c)
                self.pos += 1
                self.column += 1
            else:
                break

        value_str = "".join(value)
        lower_value = value_str.lower()

        # Check for keyword
        if lower_value in KEYWORDS:
            return Token(KEYWORDS[lower_value], value_str, start_pos, start_line, start_col)

        return Token(TokenType.IDENTIFIER, value_str, start_pos, start_line, start_col)

    def _operator(self, start_pos: int, start_line: int, start_col: int) -> Token:
        """Parse an operator or punctuation."""
        c = self.sql[self.pos]
        self.pos += 1
        self.column += 1

        # Two-character operators
        if self.pos < len(self.sql):
            c2 = self.sql[self.pos]
            two_char = c + c2

            if two_char == "!=":
                self.pos += 1
                self.column += 1
                return Token(TokenType.NE, two_char, start_pos, start_line, start_col)
            elif two_char == "<>":
                self.pos += 1
                self.column += 1
                return Token(TokenType.NE, two_char, start_pos, start_line, start_col)
            elif two_char == "<=":
                self.pos += 1
                self.column += 1
                return Token(TokenType.LE, two_char, start_pos, start_line, start_col)
            elif two_char == ">=":
                self.pos += 1
                self.column += 1
                return Token(TokenType.GE, two_char, start_pos, start_line, start_col)
            elif two_char == "||":
                self.pos += 1
                self.column += 1
                return Token(TokenType.CONCAT, two_char, start_pos, start_line, start_col)
            elif two_char == "::":
                self.pos += 1
                self.column += 1
                return Token(TokenType.DOUBLE_COLON, two_char, start_pos, start_line, start_col)

        # Single-character operators
        single_char_tokens = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "=": TokenType.EQ,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            ";": TokenType.SEMICOLON,
            ":": TokenType.COLON,
        }

        if c in single_char_tokens:
            return Token(single_char_tokens[c], c, start_pos, start_line, start_col)

        return Token(TokenType.ERROR, f"Unknown character: {c}", start_pos, start_line, start_col)


class ParseError(Exception):
    """SQL parse error."""

    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        location = ""
        if token:
            location = f" at line {token.line}, column {token.column}"
        super().__init__(f"{message}{location}")


class SQLParser:
    """Recursive descent SQL parser."""

    def __init__(self, sql: str):
        self.lexer = SQLLexer(sql)
        self.tokens = self.lexer.tokens
        self.pos = 0

    def parse(self) -> SelectStatement:
        """Parse a SELECT statement."""
        stmt = self._parse_select()
        if not self._is_at_end():
            raise ParseError(f"Unexpected token: {self._current().value}", self._current())
        return stmt

    # ========================================================================
    # Token helpers
    # ========================================================================

    def _current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def _peek(self, offset: int = 0) -> Token:
        """Peek at a token."""
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def _advance(self) -> Token:
        """Advance to next token."""
        token = self._current()
        if not self._is_at_end():
            self.pos += 1
        return token

    def _is_at_end(self) -> bool:
        """Check if at end of tokens."""
        return self._current().type == TokenType.EOF

    def _check(self, *types: TokenType) -> bool:
        """Check if current token matches any type."""
        return self._current().type in types

    def _match(self, *types: TokenType) -> bool:
        """Match and advance if current token matches."""
        if self._check(*types):
            self._advance()
            return True
        return False

    def _expect(self, token_type: TokenType, message: str) -> Token:
        """Expect a specific token type."""
        if self._check(token_type):
            return self._advance()
        raise ParseError(message, self._current())

    # ========================================================================
    # Statement parsing
    # ========================================================================

    def _parse_select(self) -> SelectStatement:
        """Parse a SELECT statement."""
        self._expect(TokenType.SELECT, "Expected SELECT")

        distinct = self._match(TokenType.DISTINCT)
        if not distinct:
            self._match(TokenType.ALL)

        # Parse select list
        select_list = self._parse_select_list()

        # FROM clause
        from_clause = None
        if self._match(TokenType.FROM):
            from_clause = self._parse_from()

        # WHERE clause
        where_clause = None
        if self._match(TokenType.WHERE):
            where_clause = WhereClause(condition=self._parse_expression())

        # GROUP BY clause
        group_by = None
        if self._match(TokenType.GROUP):
            self._expect(TokenType.BY, "Expected BY after GROUP")
            group_by = self._parse_group_by()

        # HAVING clause
        having = None
        if self._match(TokenType.HAVING):
            having = HavingClause(condition=self._parse_expression())

        # ORDER BY clause
        order_by = None
        if self._match(TokenType.ORDER):
            self._expect(TokenType.BY, "Expected BY after ORDER")
            order_by = self._parse_order_by()

        # LIMIT/OFFSET clause
        limit = None
        if self._match(TokenType.LIMIT):
            limit = self._parse_limit()

        stmt = SelectStatement(
            select_list=select_list,
            distinct=distinct,
            from_clause=from_clause,
            where_clause=where_clause,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
        )

        # Set operations (UNION, INTERSECT, EXCEPT)
        if self._check(TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT):
            set_op = self._advance()
            set_all = self._match(TokenType.ALL)
            right = self._parse_select()
            stmt.set_operation = set_op.type.name
            stmt.set_all = set_all
            stmt.right_query = right

        return stmt

    def _parse_select_list(self) -> list[Expression]:
        """Parse SELECT column list."""
        if self._match(TokenType.STAR):
            return []  # Empty list means SELECT *

        expressions = [self._parse_select_item()]
        while self._match(TokenType.COMMA):
            expressions.append(self._parse_select_item())
        return expressions

    def _parse_select_item(self) -> Expression:
        """Parse a single select item."""
        expr = self._parse_expression()

        # Check for alias
        if self._match(TokenType.AS):
            alias_token = self._expect(TokenType.IDENTIFIER, "Expected alias name")
            expr.alias = alias_token.value
        elif self._check(TokenType.IDENTIFIER):
            # Implicit alias
            alias_token = self._advance()
            expr.alias = alias_token.value

        return expr

    def _parse_from(self) -> FromClause:
        """Parse FROM clause."""
        table = self._parse_table_ref()
        joins = []

        while True:
            join_type = self._parse_join_type()
            if join_type is None:
                break

            self._expect(TokenType.JOIN, "Expected JOIN")
            join_table = self._parse_table_ref()

            condition = None
            using_columns = []

            if self._match(TokenType.ON):
                condition = self._parse_expression()
            elif self._match(TokenType.USING):
                self._expect(TokenType.LPAREN, "Expected (")
                using_columns = [self._expect(TokenType.IDENTIFIER, "Expected column").value]
                while self._match(TokenType.COMMA):
                    using_columns.append(self._expect(TokenType.IDENTIFIER, "Expected column").value)
                self._expect(TokenType.RPAREN, "Expected )")

            joins.append(JoinClause(
                join_type=join_type,
                table=join_table,
                condition=condition,
                using_columns=using_columns,
            ))

        return FromClause(table=table, joins=joins)

    def _parse_join_type(self) -> Optional[JoinType]:
        """Parse join type keywords."""
        if self._match(TokenType.INNER):
            return JoinType.INNER
        elif self._match(TokenType.LEFT):
            self._match(TokenType.OUTER)
            return JoinType.LEFT
        elif self._match(TokenType.RIGHT):
            self._match(TokenType.OUTER)
            return JoinType.RIGHT
        elif self._match(TokenType.FULL):
            self._match(TokenType.OUTER)
            return JoinType.FULL
        elif self._match(TokenType.CROSS):
            return JoinType.CROSS
        elif self._check(TokenType.JOIN):
            return JoinType.INNER
        return None

    def _parse_table_ref(self) -> TableRef:
        """Parse a table reference."""
        name_token = self._expect(TokenType.IDENTIFIER, "Expected table name")
        name = name_token.value
        schema = None

        # Check for schema.table notation
        if self._match(TokenType.DOT):
            schema = name
            name = self._expect(TokenType.IDENTIFIER, "Expected table name").value

        alias = None
        if self._match(TokenType.AS):
            alias = self._expect(TokenType.IDENTIFIER, "Expected alias").value
        elif self._check(TokenType.IDENTIFIER) and not self._check(
            TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT,
            TokenType.FULL, TokenType.CROSS, TokenType.WHERE, TokenType.GROUP,
            TokenType.HAVING, TokenType.ORDER, TokenType.LIMIT, TokenType.UNION,
            TokenType.INTERSECT, TokenType.EXCEPT, TokenType.ON
        ):
            alias = self._advance().value

        return TableRef(name=name, schema=schema, alias=alias)

    def _parse_group_by(self) -> GroupByClause:
        """Parse GROUP BY clause."""
        expressions = [self._parse_expression()]
        while self._match(TokenType.COMMA):
            expressions.append(self._parse_expression())

        with_rollup = False
        with_cube = False

        if self._match(TokenType.WITH):
            if self._match(TokenType.ROLLUP):
                with_rollup = True
            elif self._match(TokenType.CUBE):
                with_cube = True

        return GroupByClause(
            expressions=expressions,
            with_rollup=with_rollup,
            with_cube=with_cube,
        )

    def _parse_order_by(self) -> OrderByClause:
        """Parse ORDER BY clause."""
        items = [self._parse_order_item()]
        while self._match(TokenType.COMMA):
            items.append(self._parse_order_item())
        return OrderByClause(items=items)

    def _parse_order_item(self) -> OrderByItem:
        """Parse a single ORDER BY item."""
        expr = self._parse_expression()
        order = SortOrder.ASC

        if self._match(TokenType.ASC):
            order = SortOrder.ASC
        elif self._match(TokenType.DESC):
            order = SortOrder.DESC

        nulls = None
        if self._match(TokenType.NULLS):
            if self._match(TokenType.FIRST):
                nulls = NullOrdering.NULLS_FIRST
            elif self._match(TokenType.LAST):
                nulls = NullOrdering.NULLS_LAST

        return OrderByItem(expression=expr, order=order, nulls=nulls)

    def _parse_limit(self) -> LimitClause:
        """Parse LIMIT/OFFSET clause."""
        limit_val = int(self._expect(TokenType.INTEGER, "Expected limit value").value)
        offset_val = None

        if self._match(TokenType.OFFSET):
            offset_val = int(self._expect(TokenType.INTEGER, "Expected offset value").value)

        return LimitClause(limit=limit_val, offset=offset_val)

    # ========================================================================
    # Expression parsing (precedence climbing)
    # ========================================================================

    def _parse_expression(self) -> Expression:
        """Parse an expression."""
        return self._parse_or()

    def _parse_or(self) -> Expression:
        """Parse OR expressions."""
        left = self._parse_and()
        while self._match(TokenType.OR):
            right = self._parse_and()
            left = BinaryOp(operator="OR", left=left, right=right)
        return left

    def _parse_and(self) -> Expression:
        """Parse AND expressions."""
        left = self._parse_not()
        while self._match(TokenType.AND):
            right = self._parse_not()
            left = BinaryOp(operator="AND", left=left, right=right)
        return left

    def _parse_not(self) -> Expression:
        """Parse NOT expressions."""
        if self._match(TokenType.NOT):
            operand = self._parse_not()
            return UnaryOp(operator="NOT", operand=operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        """Parse comparison expressions."""
        left = self._parse_addition()

        # IS NULL / IS NOT NULL
        if self._match(TokenType.IS):
            negated = self._match(TokenType.NOT)
            self._expect(TokenType.NULL, "Expected NULL after IS")
            return IsNull(expr=left, negated=negated)

        # IN
        if self._match(TokenType.IN):
            return self._parse_in(left, negated=False)
        if self._check(TokenType.NOT) and self._peek(1).type == TokenType.IN:
            self._advance()  # NOT
            self._advance()  # IN
            return self._parse_in(left, negated=True)

        # BETWEEN
        if self._match(TokenType.BETWEEN):
            return self._parse_between(left, negated=False)
        if self._check(TokenType.NOT) and self._peek(1).type == TokenType.BETWEEN:
            self._advance()  # NOT
            self._advance()  # BETWEEN
            return self._parse_between(left, negated=True)

        # LIKE
        if self._match(TokenType.LIKE):
            return self._parse_like(left, negated=False)
        if self._check(TokenType.NOT) and self._peek(1).type == TokenType.LIKE:
            self._advance()  # NOT
            self._advance()  # LIKE
            return self._parse_like(left, negated=True)

        # Comparison operators
        op_map = {
            TokenType.EQ: "=",
            TokenType.NE: "<>",
            TokenType.LT: "<",
            TokenType.LE: "<=",
            TokenType.GT: ">",
            TokenType.GE: ">=",
        }

        for token_type, op in op_map.items():
            if self._match(token_type):
                right = self._parse_addition()
                return BinaryOp(operator=op, left=left, right=right)

        return left

    def _parse_in(self, expr: Expression, negated: bool) -> Expression:
        """Parse IN expression."""
        self._expect(TokenType.LPAREN, "Expected (")

        # Check for subquery
        if self._check(TokenType.SELECT):
            subquery = self._parse_select()
            self._expect(TokenType.RPAREN, "Expected )")
            return SubqueryExpr(query=subquery, subquery_type="in")

        # Value list
        values = [self._parse_expression()]
        while self._match(TokenType.COMMA):
            values.append(self._parse_expression())
        self._expect(TokenType.RPAREN, "Expected )")

        return InList(expr=expr, values=values, negated=negated)

    def _parse_between(self, expr: Expression, negated: bool) -> Expression:
        """Parse BETWEEN expression."""
        low = self._parse_addition()
        self._expect(TokenType.AND, "Expected AND in BETWEEN")
        high = self._parse_addition()
        return Between(expr=expr, low=low, high=high, negated=negated)

    def _parse_like(self, expr: Expression, negated: bool) -> Expression:
        """Parse LIKE expression."""
        pattern = self._parse_addition()
        escape = None
        if self._match(TokenType.ESCAPE):
            escape_token = self._expect(TokenType.STRING, "Expected escape character")
            escape = escape_token.value
        return Like(expr=expr, pattern=pattern, escape=escape, negated=negated)

    def _parse_addition(self) -> Expression:
        """Parse addition/subtraction expressions."""
        left = self._parse_multiplication()

        while self._check(TokenType.PLUS, TokenType.MINUS, TokenType.CONCAT):
            op_token = self._advance()
            op = {TokenType.PLUS: "+", TokenType.MINUS: "-", TokenType.CONCAT: "||"}[op_token.type]
            right = self._parse_multiplication()
            left = BinaryOp(operator=op, left=left, right=right)

        return left

    def _parse_multiplication(self) -> Expression:
        """Parse multiplication/division expressions."""
        left = self._parse_unary()

        while self._check(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op_token = self._advance()
            op = {TokenType.STAR: "*", TokenType.SLASH: "/", TokenType.PERCENT: "%"}[op_token.type]
            right = self._parse_unary()
            left = BinaryOp(operator=op, left=left, right=right)

        return left

    def _parse_unary(self) -> Expression:
        """Parse unary expressions."""
        if self._match(TokenType.MINUS):
            operand = self._parse_unary()
            return UnaryOp(operator="-", operand=operand)
        if self._match(TokenType.PLUS):
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        """Parse primary expressions."""
        # Literals
        if self._match(TokenType.INTEGER):
            return Literal(value=int(self._peek(-1).value), data_type=DataType.INTEGER)
        if self._match(TokenType.FLOAT):
            return Literal(value=float(self._peek(-1).value), data_type=DataType.FLOAT)
        if self._match(TokenType.STRING):
            return Literal(value=self._peek(-1).value, data_type=DataType.VARCHAR)
        if self._match(TokenType.TRUE):
            return Literal(value=True, data_type=DataType.BOOLEAN)
        if self._match(TokenType.FALSE):
            return Literal(value=False, data_type=DataType.BOOLEAN)
        if self._match(TokenType.NULL):
            return Literal(value=None, data_type=DataType.NULL)

        # CASE expression
        if self._match(TokenType.CASE):
            return self._parse_case()

        # CAST expression
        if self._match(TokenType.CAST):
            return self._parse_cast()

        # EXISTS subquery
        if self._match(TokenType.EXISTS):
            self._expect(TokenType.LPAREN, "Expected (")
            subquery = self._parse_select()
            self._expect(TokenType.RPAREN, "Expected )")
            return SubqueryExpr(query=subquery, subquery_type="exists")

        # Aggregate functions
        if self._check(TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX):
            return self._parse_aggregate()

        # Parenthesized expression or subquery
        if self._match(TokenType.LPAREN):
            if self._check(TokenType.SELECT):
                subquery = self._parse_select()
                self._expect(TokenType.RPAREN, "Expected )")
                return SubqueryExpr(query=subquery, subquery_type="scalar")
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected )")
            return expr

        # Identifier (column ref or function call)
        if self._check(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
            return self._parse_identifier_or_function()

        raise ParseError(f"Unexpected token: {self._current().value}", self._current())

    def _parse_case(self) -> Expression:
        """Parse CASE expression."""
        operand = None
        if not self._check(TokenType.WHEN):
            operand = self._parse_expression()

        when_clauses = []
        while self._match(TokenType.WHEN):
            when_cond = self._parse_expression()
            self._expect(TokenType.THEN, "Expected THEN")
            then_expr = self._parse_expression()
            when_clauses.append((when_cond, then_expr))

        else_clause = None
        if self._match(TokenType.ELSE):
            else_clause = self._parse_expression()

        self._expect(TokenType.END, "Expected END")
        return CaseExpr(operand=operand, when_clauses=when_clauses, else_clause=else_clause)

    def _parse_cast(self) -> Expression:
        """Parse CAST expression."""
        self._expect(TokenType.LPAREN, "Expected (")
        expr = self._parse_expression()
        self._expect(TokenType.AS, "Expected AS")

        type_token = self._expect(TokenType.IDENTIFIER, "Expected type name")
        type_name = type_token.value.upper()

        type_map = {
            "INT": DataType.INTEGER,
            "INTEGER": DataType.INTEGER,
            "BIGINT": DataType.BIGINT,
            "FLOAT": DataType.FLOAT,
            "DOUBLE": DataType.DOUBLE,
            "DECIMAL": DataType.DECIMAL,
            "VARCHAR": DataType.VARCHAR,
            "TEXT": DataType.TEXT,
            "BOOLEAN": DataType.BOOLEAN,
            "BOOL": DataType.BOOLEAN,
            "TIMESTAMP": DataType.TIMESTAMP,
            "DATE": DataType.DATE,
            "TIME": DataType.TIME,
        }
        target_type = type_map.get(type_name, DataType.UNKNOWN)

        self._expect(TokenType.RPAREN, "Expected )")
        return Cast(expr=expr, target_type=target_type, data_type=target_type)

    def _parse_aggregate(self) -> Expression:
        """Parse aggregate function."""
        func_token = self._advance()
        func_map = {
            TokenType.COUNT: AggregateFunction.COUNT,
            TokenType.SUM: AggregateFunction.SUM,
            TokenType.AVG: AggregateFunction.AVG,
            TokenType.MIN: AggregateFunction.MIN,
            TokenType.MAX: AggregateFunction.MAX,
        }
        func = func_map[func_token.type]

        self._expect(TokenType.LPAREN, "Expected (")
        distinct = self._match(TokenType.DISTINCT)

        args = []
        if self._match(TokenType.STAR):
            pass  # COUNT(*)
        elif not self._check(TokenType.RPAREN):
            args = [self._parse_expression()]
            while self._match(TokenType.COMMA):
                args.append(self._parse_expression())

        self._expect(TokenType.RPAREN, "Expected )")

        # Optional FILTER clause
        filter_clause = None
        if self._match(TokenType.FILTER):
            self._expect(TokenType.LPAREN, "Expected (")
            self._expect(TokenType.WHERE, "Expected WHERE")
            filter_clause = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected )")

        return AggregateExpr(function=func, args=args, distinct=distinct, filter_clause=filter_clause)

    def _parse_identifier_or_function(self) -> Expression:
        """Parse identifier (column ref) or function call."""
        name_token = self._advance()
        name = name_token.value

        # Check for function call
        if self._match(TokenType.LPAREN):
            args = []
            distinct = self._match(TokenType.DISTINCT)

            if not self._check(TokenType.RPAREN):
                args = [self._parse_expression()]
                while self._match(TokenType.COMMA):
                    args.append(self._parse_expression())

            self._expect(TokenType.RPAREN, "Expected )")
            return FunctionCall(name=name, args=args, distinct=distinct)

        # Column reference with optional table/schema
        table = None
        schema = None

        while self._match(TokenType.DOT):
            schema = table
            table = name
            name_token = self._expect(TokenType.IDENTIFIER, "Expected column name")
            name = name_token.value

        return ColumnRef(name=name, table=table, schema=schema)
