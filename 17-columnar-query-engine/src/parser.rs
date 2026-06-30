//! SQL parser and plan builder.
//!
//! Parses SQL strings into LogicalPlan nodes using the sqlparser crate.

use crate::expression::{BinaryOperator, Expression, UnaryOperator};
use crate::plan::LogicalPlan;
use crate::storage::Catalog;
use crate::types::{AggregateFunction, DataType, JoinType, Schema, SortOrder, Value};
use crate::{Error, Result};

use sqlparser::ast::{
    self, BinaryOperator as SqlBinaryOp, Expr as SqlExpr, JoinConstraint, JoinOperator,
    ObjectName, Query, Select, SelectItem, SetExpr, Statement, TableFactor, TableWithJoins,
    UnaryOperator as SqlUnaryOp, Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use std::collections::HashMap;
use std::sync::Arc;

/// SQL parser and plan builder.
pub struct SqlParser<'a> {
    catalog: &'a Catalog,
}

impl<'a> SqlParser<'a> {
    /// Create a new SQL parser with catalog reference.
    pub fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }

    /// Parse and plan a SQL query string.
    pub fn parse(&self, sql: &str) -> Result<LogicalPlan> {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|e| Error::Parse(format!("SQL parse error: {}", e)))?;

        if statements.is_empty() {
            return Err(Error::Parse("Empty SQL statement".to_string()));
        }

        if statements.len() > 1 {
            return Err(Error::Parse("Multiple statements not supported".to_string()));
        }

        self.plan_statement(&statements[0])
    }

    fn plan_statement(&self, stmt: &Statement) -> Result<LogicalPlan> {
        match stmt {
            Statement::Query(query) => self.plan_query(query),
            _ => Err(Error::Parse(format!(
                "Unsupported statement type: {:?}",
                stmt
            ))),
        }
    }

    fn plan_query(&self, query: &Query) -> Result<LogicalPlan> {
        // Handle CTEs (WITH clause) - not supported yet
        if !query.with.is_none() {
            return Err(Error::Parse("WITH clause not supported".to_string()));
        }

        let mut plan = self.plan_set_expr(&query.body)?;

        // ORDER BY
        if !query.order_by.is_empty() {
            let order_by: Vec<(Expression, SortOrder, bool)> = query
                .order_by
                .iter()
                .map(|o| {
                    let expr = self.plan_expr(&o.expr)?;
                    let order = if o.asc.unwrap_or(true) {
                        SortOrder::Ascending
                    } else {
                        SortOrder::Descending
                    };
                    let nulls_first = o.nulls_first.unwrap_or(order == SortOrder::Ascending);
                    Ok((expr, order, nulls_first))
                })
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalPlan::Sort {
                input: Arc::new(plan),
                order_by,
            };
        }

        // LIMIT / OFFSET
        if query.limit.is_some() || query.offset.is_some() {
            let limit = query
                .limit
                .as_ref()
                .map(|l| self.expr_to_usize(l))
                .transpose()?
                .unwrap_or(usize::MAX);
            let offset = query
                .offset
                .as_ref()
                .map(|o| self.expr_to_usize(&o.value))
                .transpose()?
                .unwrap_or(0);

            plan = LogicalPlan::Limit {
                input: Arc::new(plan),
                limit,
                offset,
            };
        }

        Ok(plan)
    }

    fn expr_to_usize(&self, expr: &SqlExpr) -> Result<usize> {
        match expr {
            SqlExpr::Value(SqlValue::Number(n, _)) => n
                .parse()
                .map_err(|_| Error::Parse(format!("Invalid number: {}", n))),
            _ => Err(Error::Parse("Expected numeric literal".to_string())),
        }
    }

    fn plan_set_expr(&self, set_expr: &SetExpr) -> Result<LogicalPlan> {
        match set_expr {
            SetExpr::Select(select) => self.plan_select(select),
            SetExpr::Query(query) => self.plan_query(query),
            SetExpr::SetOperation {
                op,
                set_quantifier,
                left,
                right,
                ..
            } => {
                let left_plan = self.plan_set_expr(left)?;
                let right_plan = self.plan_set_expr(right)?;

                let all = matches!(set_quantifier, ast::SetQuantifier::All);

                match op {
                    ast::SetOperator::Union => Ok(LogicalPlan::Union {
                        left: Arc::new(left_plan),
                        right: Arc::new(right_plan),
                        all,
                    }),
                    _ => Err(Error::Parse(format!("Unsupported set operation: {:?}", op))),
                }
            }
            SetExpr::Values(values) => {
                let rows: Vec<Vec<Expression>> = values
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|e| self.plan_expr(e)).collect())
                    .collect::<Result<Vec<_>>>()?;

                if rows.is_empty() {
                    return Err(Error::Parse("VALUES requires at least one row".to_string()));
                }

                // Infer schema from first row
                let schema = Schema::from_pairs(
                    rows[0]
                        .iter()
                        .enumerate()
                        .map(|(i, _)| (format!("column{}", i), DataType::String))
                        .collect(),
                );

                Ok(LogicalPlan::Values {
                    values: rows,
                    schema,
                })
            }
            _ => Err(Error::Parse(format!(
                "Unsupported set expression: {:?}",
                set_expr
            ))),
        }
    }

    fn plan_select(&self, select: &Select) -> Result<LogicalPlan> {
        // Start with FROM clause
        let mut plan = if select.from.is_empty() {
            // No FROM clause - use empty schema
            LogicalPlan::Empty {
                schema: Schema::empty(),
            }
        } else {
            self.plan_from(&select.from)?
        };

        // WHERE clause
        if let Some(selection) = &select.selection {
            let predicate = self.plan_expr(selection)?;
            plan = LogicalPlan::Filter {
                input: Arc::new(plan),
                predicate,
            };
        }

        // GROUP BY and aggregates
        let has_aggregates = self.has_aggregates(&select.projection);
        let has_group_by = matches!(&select.group_by, ast::GroupByExpr::Expressions(exprs) if !exprs.is_empty());
        if has_group_by || has_aggregates {
            let group_by: Vec<Expression> = match &select.group_by {
                ast::GroupByExpr::Expressions(exprs) => {
                    exprs.iter().map(|e| self.plan_expr(e)).collect::<Result<_>>()?
                }
                ast::GroupByExpr::All => {
                    return Err(Error::Parse("GROUP BY ALL not supported".to_string()));
                }
            };

            let aggregates = self.extract_aggregates(&select.projection)?;
            let schema = self.build_aggregate_schema(&group_by, &aggregates, plan.schema())?;

            plan = LogicalPlan::Aggregate {
                input: Arc::new(plan),
                group_by,
                aggregates,
                schema,
            };
        }

        // HAVING clause
        if let Some(having) = &select.having {
            let predicate = self.plan_expr(having)?;
            plan = LogicalPlan::Filter {
                input: Arc::new(plan),
                predicate,
            };
        }

        // SELECT projection
        let (expressions, schema) = self.plan_projection(&select.projection, plan.schema())?;

        plan = LogicalPlan::Projection {
            input: Arc::new(plan),
            expressions,
            schema,
        };

        // DISTINCT
        if select.distinct.is_some() {
            plan = LogicalPlan::Distinct {
                input: Arc::new(plan),
            };
        }

        Ok(plan)
    }

    fn plan_from(&self, from: &[TableWithJoins]) -> Result<LogicalPlan> {
        if from.is_empty() {
            return Err(Error::Parse("FROM clause is empty".to_string()));
        }

        let mut plan = self.plan_table_with_joins(&from[0])?;

        // Handle multiple tables in FROM (implicit cross join)
        for table in from.iter().skip(1) {
            let right = self.plan_table_with_joins(table)?;
            let left_schema = plan.schema().clone();
            let right_schema = right.schema().clone();

            plan = LogicalPlan::Join {
                left: Arc::new(plan),
                right: Arc::new(right),
                join_type: JoinType::Cross,
                condition: None,
                schema: left_schema.merge(&right_schema),
            };
        }

        Ok(plan)
    }

    fn plan_table_with_joins(&self, table: &TableWithJoins) -> Result<LogicalPlan> {
        let mut plan = self.plan_table_factor(&table.relation)?;

        for join in &table.joins {
            let right = self.plan_table_factor(&join.relation)?;
            let (join_type, condition) = self.plan_join_operator(&join.join_operator)?;

            let left_schema = plan.schema().clone();
            let right_schema = right.schema().clone();

            plan = LogicalPlan::Join {
                left: Arc::new(plan),
                right: Arc::new(right),
                join_type,
                condition,
                schema: left_schema.merge(&right_schema),
            };
        }

        Ok(plan)
    }

    fn plan_table_factor(&self, factor: &TableFactor) -> Result<LogicalPlan> {
        match factor {
            TableFactor::Table { name, alias, .. } => {
                let table_name = self.object_name_to_string(name);
                let schema = self.catalog.get_table_schema(&table_name)?;

                Ok(LogicalPlan::Scan {
                    table_name: alias
                        .as_ref()
                        .map(|a| a.name.value.clone())
                        .unwrap_or(table_name),
                    projection: None,
                    filter: None,
                    schema,
                })
            }
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                let plan = self.plan_query(subquery)?;
                let alias_name = alias
                    .as_ref()
                    .map(|a| a.name.value.clone())
                    .unwrap_or_else(|| "subquery".to_string());

                Ok(LogicalPlan::Subquery {
                    input: Arc::new(plan),
                    alias: alias_name,
                })
            }
            TableFactor::NestedJoin { table_with_joins, .. } => {
                self.plan_table_with_joins(table_with_joins)
            }
            _ => Err(Error::Parse(format!(
                "Unsupported table factor: {:?}",
                factor
            ))),
        }
    }

    fn plan_join_operator(
        &self,
        op: &JoinOperator,
    ) -> Result<(JoinType, Option<Expression>)> {
        match op {
            JoinOperator::Inner(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Inner, condition))
            }
            JoinOperator::LeftOuter(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Left, condition))
            }
            JoinOperator::RightOuter(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Right, condition))
            }
            JoinOperator::FullOuter(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Full, condition))
            }
            JoinOperator::CrossJoin => Ok((JoinType::Cross, None)),
            JoinOperator::LeftSemi(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Semi, condition))
            }
            JoinOperator::LeftAnti(constraint) => {
                let condition = self.plan_join_constraint(constraint)?;
                Ok((JoinType::Anti, condition))
            }
            _ => Err(Error::Parse(format!("Unsupported join type: {:?}", op))),
        }
    }

    fn plan_join_constraint(&self, constraint: &JoinConstraint) -> Result<Option<Expression>> {
        match constraint {
            JoinConstraint::On(expr) => Ok(Some(self.plan_expr(expr)?)),
            JoinConstraint::None => Ok(None),
            JoinConstraint::Natural => Err(Error::Parse("NATURAL JOIN not supported".to_string())),
            JoinConstraint::Using(_) => Err(Error::Parse("USING clause not supported".to_string())),
        }
    }

    fn plan_expr(&self, expr: &SqlExpr) -> Result<Expression> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(Expression::ColumnName(ident.value.clone())),

            SqlExpr::CompoundIdentifier(parts) => {
                let name = parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                Ok(Expression::ColumnName(name))
            }

            SqlExpr::Value(value) => Ok(Expression::Literal(self.plan_value(value)?)),

            SqlExpr::BinaryOp { left, op, right } => {
                let left_expr = self.plan_expr(left)?;
                let right_expr = self.plan_expr(right)?;
                let binary_op = self.plan_binary_op(op)?;

                Ok(Expression::BinaryOp {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }

            SqlExpr::UnaryOp { op, expr } => {
                let operand = self.plan_expr(expr)?;
                let unary_op = self.plan_unary_op(op)?;

                Ok(Expression::UnaryOp {
                    op: unary_op,
                    expr: Box::new(operand),
                })
            }

            SqlExpr::Function(func) => self.plan_function(func),

            SqlExpr::Nested(inner) => self.plan_expr(inner),

            SqlExpr::IsNull(expr) => Ok(Expression::IsNull(Box::new(self.plan_expr(expr)?))),

            SqlExpr::IsNotNull(expr) => Ok(Expression::IsNotNull(Box::new(self.plan_expr(expr)?))),

            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let expr = self.plan_expr(expr)?;
                let list: Vec<Expression> =
                    list.iter().map(|e| self.plan_expr(e)).collect::<Result<_>>()?;

                Ok(Expression::InList {
                    expr: Box::new(expr),
                    list,
                    negated: *negated,
                })
            }

            SqlExpr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let expr = self.plan_expr(expr)?;
                let low = self.plan_expr(low)?;
                let high = self.plan_expr(high)?;

                Ok(Expression::Between {
                    expr: Box::new(expr),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated: *negated,
                })
            }

            SqlExpr::Like {
                expr,
                pattern,
                negated,
                ..
            } => {
                let expr = self.plan_expr(expr)?;
                let pattern = match self.plan_expr(pattern)? {
                    Expression::Literal(Value::String(s)) => s,
                    _ => return Err(Error::Parse("LIKE pattern must be a string".to_string())),
                };

                Ok(Expression::Like {
                    expr: Box::new(expr),
                    pattern,
                    negated: *negated,
                })
            }

            SqlExpr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => {
                let operand = operand
                    .as_ref()
                    .map(|e| self.plan_expr(e))
                    .transpose()?
                    .map(Box::new);

                let when_clauses: Vec<(Expression, Expression)> = conditions
                    .iter()
                    .zip(results.iter())
                    .map(|(c, r)| Ok((self.plan_expr(c)?, self.plan_expr(r)?)))
                    .collect::<Result<_>>()?;

                let else_result = else_result
                    .as_ref()
                    .map(|e| self.plan_expr(e))
                    .transpose()?
                    .map(Box::new);

                Ok(Expression::Case {
                    operand,
                    when_clauses,
                    else_result,
                })
            }

            SqlExpr::Cast { expr, data_type, .. } => {
                let expr = self.plan_expr(expr)?;
                let target_type = self.plan_data_type(data_type)?;

                Ok(Expression::Cast {
                    expr: Box::new(expr),
                    target_type,
                })
            }

            _ => Err(Error::Parse(format!("Unsupported expression: {:?}", expr))),
        }
    }

    fn plan_value(&self, value: &SqlValue) -> Result<Value> {
        match value {
            SqlValue::Number(n, _) => {
                if n.contains('.') {
                    n.parse::<f64>()
                        .map(Value::Float64)
                        .map_err(|_| Error::Parse(format!("Invalid float: {}", n)))
                } else {
                    n.parse::<i64>()
                        .map(Value::Int64)
                        .map_err(|_| Error::Parse(format!("Invalid integer: {}", n)))
                }
            }
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                Ok(Value::String(s.clone()))
            }
            SqlValue::Boolean(b) => Ok(Value::Boolean(*b)),
            SqlValue::Null => Ok(Value::Null),
            _ => Err(Error::Parse(format!("Unsupported value: {:?}", value))),
        }
    }

    fn plan_binary_op(&self, op: &SqlBinaryOp) -> Result<BinaryOperator> {
        match op {
            SqlBinaryOp::Plus => Ok(BinaryOperator::Add),
            SqlBinaryOp::Minus => Ok(BinaryOperator::Subtract),
            SqlBinaryOp::Multiply => Ok(BinaryOperator::Multiply),
            SqlBinaryOp::Divide => Ok(BinaryOperator::Divide),
            SqlBinaryOp::Modulo => Ok(BinaryOperator::Modulo),
            SqlBinaryOp::Eq => Ok(BinaryOperator::Equal),
            SqlBinaryOp::NotEq => Ok(BinaryOperator::NotEqual),
            SqlBinaryOp::Lt => Ok(BinaryOperator::LessThan),
            SqlBinaryOp::LtEq => Ok(BinaryOperator::LessThanOrEqual),
            SqlBinaryOp::Gt => Ok(BinaryOperator::GreaterThan),
            SqlBinaryOp::GtEq => Ok(BinaryOperator::GreaterThanOrEqual),
            SqlBinaryOp::And => Ok(BinaryOperator::And),
            SqlBinaryOp::Or => Ok(BinaryOperator::Or),
            _ => Err(Error::Parse(format!("Unsupported binary operator: {:?}", op))),
        }
    }

    fn plan_unary_op(&self, op: &SqlUnaryOp) -> Result<UnaryOperator> {
        match op {
            SqlUnaryOp::Not => Ok(UnaryOperator::Not),
            SqlUnaryOp::Minus => Ok(UnaryOperator::Negate),
            SqlUnaryOp::Plus => Ok(UnaryOperator::Plus),
            _ => Err(Error::Parse(format!("Unsupported unary operator: {:?}", op))),
        }
    }

    fn plan_function(&self, func: &ast::Function) -> Result<Expression> {
        let name = self.object_name_to_string(&func.name).to_uppercase();

        // Check for aggregate functions
        if let Some(agg_func) = self.try_aggregate_function(&name) {
            let args: Vec<Expression> = func
                .args
                .iter()
                .map(|arg| match arg {
                    ast::FunctionArg::Named { arg, .. } => self.plan_function_arg_expr(arg),
                    ast::FunctionArg::Unnamed(arg) => self.plan_function_arg_expr(arg),
                })
                .collect::<Result<_>>()?;

            let expr = args.into_iter().next().unwrap_or(Expression::Wildcard);
            let distinct = func.distinct;

            return Ok(Expression::Aggregate {
                func: agg_func,
                expr: Box::new(expr),
                distinct,
            });
        }

        // Regular function
        let args: Vec<Expression> = func
            .args
            .iter()
            .map(|arg| match arg {
                ast::FunctionArg::Named { arg, .. } => self.plan_function_arg_expr(&arg),
                ast::FunctionArg::Unnamed(arg) => self.plan_function_arg_expr(&arg),
            })
            .collect::<Result<_>>()?;

        Ok(Expression::Function { name, args })
    }

    fn plan_function_arg_expr(&self, arg: &ast::FunctionArgExpr) -> Result<Expression> {
        match arg {
            ast::FunctionArgExpr::Expr(expr) => self.plan_expr(expr),
            ast::FunctionArgExpr::Wildcard => Ok(Expression::Wildcard),
            ast::FunctionArgExpr::QualifiedWildcard(_) => Ok(Expression::Wildcard),
        }
    }

    fn try_aggregate_function(&self, name: &str) -> Option<AggregateFunction> {
        match name {
            "COUNT" => Some(AggregateFunction::Count),
            "SUM" => Some(AggregateFunction::Sum),
            "AVG" => Some(AggregateFunction::Avg),
            "MIN" => Some(AggregateFunction::Min),
            "MAX" => Some(AggregateFunction::Max),
            "FIRST" => Some(AggregateFunction::First),
            "LAST" => Some(AggregateFunction::Last),
            _ => None,
        }
    }

    fn plan_data_type(&self, dt: &ast::DataType) -> Result<DataType> {
        match dt {
            ast::DataType::Boolean => Ok(DataType::Boolean),
            ast::DataType::TinyInt(_) => Ok(DataType::Int8),
            ast::DataType::SmallInt(_) => Ok(DataType::Int16),
            ast::DataType::Int(_) | ast::DataType::Integer(_) => Ok(DataType::Int32),
            ast::DataType::BigInt(_) => Ok(DataType::Int64),
            ast::DataType::Real | ast::DataType::Float(_) => Ok(DataType::Float32),
            ast::DataType::Double | ast::DataType::DoublePrecision => Ok(DataType::Float64),
            ast::DataType::Varchar(_)
            | ast::DataType::Char(_)
            | ast::DataType::Text
            | ast::DataType::String(_) => Ok(DataType::String),
            ast::DataType::Date => Ok(DataType::Date),
            ast::DataType::Timestamp(_, _) => Ok(DataType::Timestamp),
            ast::DataType::Decimal(info) => {
                let (precision, scale) = match info {
                    ast::ExactNumberInfo::None => (18, 0),
                    ast::ExactNumberInfo::Precision(p) => (*p as u8, 0),
                    ast::ExactNumberInfo::PrecisionAndScale(p, s) => (*p as u8, *s as u8),
                };
                Ok(DataType::Decimal { precision, scale })
            }
            _ => Err(Error::Parse(format!("Unsupported data type: {:?}", dt))),
        }
    }

    fn has_aggregates(&self, items: &[SelectItem]) -> bool {
        items.iter().any(|item| match item {
            SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                self.expr_has_aggregate(expr)
            }
            _ => false,
        })
    }

    fn expr_has_aggregate(&self, expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::Function(func) => {
                let name = self.object_name_to_string(&func.name).to_uppercase();
                self.try_aggregate_function(&name).is_some()
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                self.expr_has_aggregate(left) || self.expr_has_aggregate(right)
            }
            SqlExpr::UnaryOp { expr, .. } => self.expr_has_aggregate(expr),
            SqlExpr::Nested(inner) => self.expr_has_aggregate(inner),
            _ => false,
        }
    }

    fn extract_aggregates(
        &self,
        items: &[SelectItem],
    ) -> Result<Vec<(AggregateFunction, Expression, String)>> {
        let mut aggregates = Vec::new();

        for item in items {
            match item {
                SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                    self.collect_aggregates(expr, &mut aggregates)?;
                }
                _ => {}
            }
        }

        Ok(aggregates)
    }

    fn collect_aggregates(
        &self,
        expr: &SqlExpr,
        aggregates: &mut Vec<(AggregateFunction, Expression, String)>,
    ) -> Result<()> {
        match expr {
            SqlExpr::Function(func) => {
                let name = self.object_name_to_string(&func.name).to_uppercase();
                if let Some(agg_func) = self.try_aggregate_function(&name) {
                    let args: Vec<Expression> = func
                        .args
                        .iter()
                        .map(|arg| match arg {
                            ast::FunctionArg::Named { arg, .. } => self.plan_function_arg_expr(&arg),
                            ast::FunctionArg::Unnamed(arg) => self.plan_function_arg_expr(&arg),
                        })
                        .collect::<Result<_>>()?;

                    let arg_expr = args.into_iter().next().unwrap_or(Expression::Wildcard);
                    let alias = format!("{}({})", name.to_lowercase(), "expr");
                    aggregates.push((agg_func, arg_expr, alias));
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                self.collect_aggregates(left, aggregates)?;
                self.collect_aggregates(right, aggregates)?;
            }
            SqlExpr::UnaryOp { expr, .. } => {
                self.collect_aggregates(expr, aggregates)?;
            }
            SqlExpr::Nested(inner) => {
                self.collect_aggregates(inner, aggregates)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn plan_projection(
        &self,
        items: &[SelectItem],
        input_schema: &Schema,
    ) -> Result<(Vec<Expression>, Schema)> {
        let mut expressions = Vec::new();
        let mut columns = Vec::new();

        for item in items {
            match item {
                SelectItem::UnnamedExpr(expr) => {
                    let planned = self.plan_expr(expr)?;
                    let name = self.expr_name(expr);
                    let dtype = self.infer_type(&planned, input_schema)?;
                    expressions.push(planned);
                    columns.push((name, dtype));
                }
                SelectItem::ExprWithAlias { expr, alias } => {
                    let planned = self.plan_expr(expr)?;
                    let dtype = self.infer_type(&planned, input_schema)?;
                    expressions.push(planned);
                    columns.push((alias.value.clone(), dtype));
                }
                SelectItem::Wildcard(_) => {
                    for (name, dtype) in input_schema.columns() {
                        expressions.push(Expression::ColumnName(name.clone()));
                        columns.push((name.clone(), dtype.clone()));
                    }
                }
                SelectItem::QualifiedWildcard(name, _) => {
                    let prefix = self.object_name_to_string(name);
                    for (col_name, dtype) in input_schema.columns() {
                        if col_name.starts_with(&format!("{}.", prefix)) {
                            expressions.push(Expression::ColumnName(col_name.clone()));
                            columns.push((col_name.clone(), dtype.clone()));
                        }
                    }
                }
            }
        }

        Ok((expressions, Schema::from_pairs(columns)))
    }

    fn build_aggregate_schema(
        &self,
        group_by: &[Expression],
        aggregates: &[(AggregateFunction, Expression, String)],
        input_schema: &Schema,
    ) -> Result<Schema> {
        let mut columns = Vec::new();

        // Group by columns
        for expr in group_by {
            let name = match expr {
                Expression::ColumnName(n) => n.clone(),
                Expression::ColumnRef(i) => input_schema
                    .columns()
                    .get(*i)
                    .map(|(n, _)| n.clone())
                    .unwrap_or_else(|| format!("col{}", i)),
                _ => "group_key".to_string(),
            };
            let dtype = self.infer_type(expr, input_schema)?;
            columns.push((name, dtype));
        }

        // Aggregate columns
        for (func, _, alias) in aggregates {
            let dtype = match func {
                AggregateFunction::Count => DataType::Int64,
                AggregateFunction::Sum | AggregateFunction::Avg => DataType::Float64,
                _ => DataType::Float64,
            };
            columns.push((alias.clone(), dtype));
        }

        Ok(Schema::from_pairs(columns))
    }

    fn infer_type(&self, expr: &Expression, schema: &Schema) -> Result<DataType> {
        match expr {
            Expression::ColumnRef(i) => schema
                .columns()
                .get(*i)
                .map(|(_, dt)| dt.clone())
                .ok_or_else(|| Error::ColumnNotFound(format!("Column index {}", i))),
            Expression::ColumnName(name) => schema
                .get_column_type(name)
                .ok_or_else(|| Error::ColumnNotFound(name.clone())),
            Expression::Literal(v) => Ok(v.data_type()),
            Expression::BinaryOp { left, op, right } => {
                let left_type = self.infer_type(left, schema)?;
                let right_type = self.infer_type(right, schema)?;

                match op {
                    BinaryOperator::And | BinaryOperator::Or => Ok(DataType::Boolean),
                    BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::LessThan
                    | BinaryOperator::LessThanOrEqual
                    | BinaryOperator::GreaterThan
                    | BinaryOperator::GreaterThanOrEqual => Ok(DataType::Boolean),
                    _ => {
                        // Numeric operations
                        if left_type.is_numeric() && right_type.is_numeric() {
                            Ok(DataType::Float64)
                        } else {
                            Ok(left_type)
                        }
                    }
                }
            }
            Expression::Aggregate { func, .. } => match func {
                AggregateFunction::Count => Ok(DataType::Int64),
                _ => Ok(DataType::Float64),
            },
            Expression::Cast { target_type, .. } => Ok(target_type.clone()),
            Expression::IsNull(_) | Expression::IsNotNull(_) => Ok(DataType::Boolean),
            Expression::InList { .. } | Expression::Between { .. } | Expression::Like { .. } => {
                Ok(DataType::Boolean)
            }
            _ => Ok(DataType::String),
        }
    }

    fn expr_name(&self, expr: &SqlExpr) -> String {
        match expr {
            SqlExpr::Identifier(ident) => ident.value.clone(),
            SqlExpr::CompoundIdentifier(parts) => parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join("."),
            SqlExpr::Function(func) => self.object_name_to_string(&func.name),
            _ => "expr".to_string(),
        }
    }

    fn object_name_to_string(&self, name: &ObjectName) -> String {
        name.0.iter().map(|p| p.value.as_str()).collect::<Vec<_>>().join(".")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Catalog;

    fn create_test_catalog() -> Catalog {
        let mut catalog = Catalog::new();

        // Add test tables
        let users_schema = Schema::from_pairs(vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int32),
        ]);
        catalog.register_schema("users", users_schema);

        let orders_schema = Schema::from_pairs(vec![
            ("id".to_string(), DataType::Int64),
            ("user_id".to_string(), DataType::Int64),
            ("amount".to_string(), DataType::Float64),
        ]);
        catalog.register_schema("orders", orders_schema);

        catalog
    }

    #[test]
    fn test_parse_simple_select() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser.parse("SELECT id, name FROM users").unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_select_with_where() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser.parse("SELECT * FROM users WHERE age > 18").unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_select_with_aggregate() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser.parse("SELECT COUNT(*) FROM users").unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_select_with_group_by() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT name, COUNT(*) FROM users GROUP BY name")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_select_with_order_by() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT * FROM users ORDER BY age DESC")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Limit { .. } | LogicalPlan::Sort { .. } | LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_select_with_limit() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser.parse("SELECT * FROM users LIMIT 10").unwrap();
        assert!(matches!(plan, LogicalPlan::Limit { .. }));
    }

    #[test]
    fn test_parse_join() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_left_join() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_expressions() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT id, age + 1 AS next_age FROM users WHERE age BETWEEN 18 AND 65")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_case_expression() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT CASE WHEN age < 18 THEN 'minor' ELSE 'adult' END FROM users")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_in_list() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT * FROM users WHERE id IN (1, 2, 3)")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_like() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT * FROM users WHERE name LIKE 'A%'")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_parse_union() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT id FROM users UNION ALL SELECT id FROM orders")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Union { .. }));
    }

    #[test]
    fn test_parse_distinct() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser.parse("SELECT DISTINCT name FROM users").unwrap();
        // DISTINCT wraps the projection in a Distinct node
        assert!(matches!(plan, LogicalPlan::Distinct { .. }));
    }

    #[test]
    fn test_parse_multiple_aggregates() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let plan = parser
            .parse("SELECT COUNT(*), SUM(age), AVG(age), MIN(age), MAX(age) FROM users")
            .unwrap();
        assert!(matches!(plan, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_invalid_sql() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let result = parser.parse("SELEC * FROM users");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_sql() {
        let catalog = create_test_catalog();
        let parser = SqlParser::new(&catalog);

        let result = parser.parse("");
        assert!(result.is_err());
    }
}
