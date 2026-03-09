# Top-Down Walkthrough: How `#[gen_exec]` processes LW

## Naming conventions

There are three layers of types in play. To avoid confusion, they are named distinctly:

| Layer | Types | Where |
|---|---|---|
| **syn** (Rust parser) | `syn::Stmt`, `syn::Expr`, `syn::ExprCall`, `syn::ExprStruct`, `syn::ExprPath` | syn crate |
| **Our AST** (instruction semantics) | `AstExpr`, `AstStmt` | `tracer/src/ast.rs` |
| **Backend functions** | `generate_ast_stmt`, `generate_ast_expr`, `generate_ast_cast` | `rust_backend.rs`, `lean_backend.rs` |

The proc macro receives Rust source as `syn` types. It pattern-matches on them
to recognize our AST DSL nodes (`WriteRd`, `Add`, `Rs1`, `Cast`, etc.) and
generates backend-specific output.

The `AstExpr` / `AstStmt` enums in `ast.rs` are the canonical type definitions.
They are not used at proc-macro time — the proc macro works on `syn::Expr` trees
and identifies AST nodes by their string names.

## How the DSL maps to syn types

| DSL syntax | syn type | Example |
|---|---|---|
| Bare identifier | `syn::Expr::Path` | `Rs1`, `Imm`, `Pc`, `W32`, `Signed` |
| Function call | `syn::Expr::Call` | `Add(Rs1, Imm)`, `Load(W32, addr)`, `WriteRd(expr)` |
| Struct literal | `syn::Expr::Struct` | `Cast { from: W32, to: W64, sign: Signed, expr: ... }` |
| Array literal | `syn::Expr::Array` | `[stmt1, stmt2, ...]` inside `Seq([ ... ])` |
| Integer literal | `syn::Expr::Lit` | `42`, `0` |
| Negation | `syn::Expr::Unary` | `-1`, `-8` |

## Source

In `tracer/src/instruction/lw.rs:47-52`:

```rust
#[tracer_macros::gen_exec]
impl LW {
    fn ast() -> Stmt {
        WriteRd(Cast { from: W32, to: W64, sign: Signed, expr: Load(W32, Add(Rs1, Imm)) })
    }
}
```

## AST as tree

```
WriteRd                              ← AstStmt::WriteRd
└── Cast { from: W32, to: W64,      ← AstExpr::Cast
         sign: Signed }
    └── Load(W32)                    ← AstExpr::Load
        └── Add                      ← AstExpr::Add
            ├── Rs1                  ← AstExpr::Rs1
            └── Imm                  ← AstExpr::Imm
```

---

## Step 1: Compiler invokes the proc macro

The compiler sees `#[tracer_macros::gen_exec]` on the impl block and calls
`gen_exec` (`lib.rs:20`) with the entire `impl LW { ... }` as a token stream.

## Step 2: Parse the impl block and find `fn ast()`

`gen_exec` parses the token stream into a `syn::ItemImpl` (`lib.rs:21`).

It searches the impl block's methods for one named `"ast"` (`lib.rs:24-27`):

```rust
let ast_fn = impl_block.items.iter().find_map(|item| {
    if let ImplItem::Fn(method) = item {
        if method.sig.ident == "ast" {
            return Some(method.clone());
        }
    }
    None
});
```

## Step 3: Extract the body block

It takes the body of `fn ast()` (`lib.rs:43`):

```rust
let body = &ast_fn.block;
```

`body` is a `syn::Block` — the `{ ... }` containing the AST expression.

## Step 4: `generate_exec_body` unwraps the `syn::Block`

Passed to `rust_backend::generate_exec_body(body)` (`lib.rs:44`).

Inside (`rust_backend.rs:14-26`), it extracts the last `syn::Stmt` from the
block, then unwraps it to get the inner `syn::Expr`:

```rust
let last = stmts.last().unwrap();        // syn::Stmt
let ret_expr = match last {
    syn::Stmt::Expr(expr, _) => expr,    // syn::Expr
    ...
};
generate_ast_stmt(ret_expr)              // enter AST-level processing
```

`syn::Stmt::Expr(expr, None)` is syn's representation of a return expression
(no semicolon). The inner `expr` is a `syn::Expr` — specifically a
`syn::Expr::Call` because `WriteRd(...)` is function-call syntax.

## Step 5: `generate_ast_stmt` — `WriteRd`

`generate_ast_stmt` (`rust_backend.rs:30`) receives the `syn::Expr` for:

```
WriteRd(Cast { from: W32, to: W64, sign: Signed, expr: Load(W32, Add(Rs1, Imm)) })
```

It matches `syn::Expr::Call`, extracts function name `"WriteRd"` (`rust_backend.rs:35`).

`WriteRd` has 1 argument. It calls `generate_ast_expr(&call.args[0])`
(`rust_backend.rs:39`) on that argument — the `Cast { ... }` struct.

This crosses from **AstStmt** processing into **AstExpr** processing.

## Step 6: `generate_ast_expr` — `Cast`

`generate_ast_expr` (`rust_backend.rs:141`) receives the `syn::Expr` for:

```
Cast { from: W32, to: W64, sign: Signed, expr: Load(W32, Add(Rs1, Imm)) }
```

It matches `syn::Expr::Struct` with name `"Cast"` (`rust_backend.rs:298`),
delegates to `generate_ast_cast` (`rust_backend.rs:327`).

`generate_ast_cast` extracts the fields `from`, `to`, `sign`, `expr` and
recurses into `generate_ast_expr` on the `expr` field (the `Load(...)`).

## Step 7: `generate_ast_expr` — `Load`

Receives:

```
Load(W32, Add(Rs1, Imm))
```

Matches `syn::Expr::Call` with name `"Load"` (`rust_backend.rs:220`).
First arg is width `W32`, second is the address expression.

Recurses into `generate_ast_expr` on the address (`Add(Rs1, Imm)`).

## Step 8: `generate_ast_expr` — `Add`

Receives:

```
Add(Rs1, Imm)
```

Matches `syn::Expr::Call` with name `"Add"` (`rust_backend.rs:175`).
Calls `binary_op` (`rust_backend.rs:375`) which recurses into both args.

## Step 9: `generate_ast_expr` — `Rs1`

Matches `syn::Expr::Path` with name `"Rs1"` (`rust_backend.rs:147`).

- **Rust**: `cpu.x[self.operands.rs1 as usize]`
- **Lean** (`lean_backend.rs:283`): `read rs1 s.reg`

## Step 10: `generate_ast_expr` — `Imm`

Matches `syn::Expr::Path` with name `"Imm"` (`rust_backend.rs:149`).

- **Rust**: `self.operands.imm`
- **Lean** (`lean_backend.rs:285`): `imm.signExtend 64`
