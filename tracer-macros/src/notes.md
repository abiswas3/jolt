# AST Parsing & Code Generation Walkthrough

## How the DSL maps to syn expression types

The AST DSL is written in Rust syntax. The proc macro parses it using `syn` and
pattern-matches on the expression type to determine what AST node it is:

| DSL syntax | syn type | Example |
|---|---|---|
| Bare identifier | `Expr::Path` | `Rs1`, `Imm`, `Pc`, `W32`, `Signed` |
| Function call | `Expr::Call` | `Add(Rs1, Imm)`, `Load(W32, addr)`, `WriteRd(expr)` |
| Struct literal | `Expr::Struct` | `Cast { from: W32, to: W64, sign: Signed, expr: ... }` |
| Array literal | `Expr::Array` | `[stmt1, stmt2, ...]` inside `Seq([ ... ])` |
| Integer literal | `Expr::Lit` | `42`, `0` |
| Negation | `Expr::Unary` | `-1`, `-8` |

Terminals (`Rs1`, `Imm`) are paths — just a name, no arguments.
Operations (`Add`, `Load`, `WriteRd`) are calls — a name with parenthesized arguments.
`Cast` uses struct syntax because it has named fields (`from`, `to`, `sign`, `expr`).

---

## Example: LW (Load Word)

### AST

```
WriteRd(Cast { from: W32, to: W64, sign: Signed, expr: Load(W32, Add(Rs1, Imm)) })
```

### AST as tree

```
WriteRd
└── Cast { from: W32, to: W64, sign: Signed }
    └── Load(W32)
        └── Add
            ├── Rs1
            └── Imm
```

### Step 1: `Rs1`

```
WriteRd
└── Cast { from: W32, to: W64, sign: Signed }
    └── Load(W32)
        └── Add
            ├── Rs1  ← here
            └── Imm
```

Parser sees `Expr::Path` with identifier `"Rs1"`.

- **Rust** (`rust_backend.rs:147`): `cpu.x[self.operands.rs1 as usize]`
- **Lean** (`lean_backend.rs:283`): `read rs1 s.reg`

### Step 2: `Imm`

```
WriteRd
└── Cast { from: W32, to: W64, sign: Signed }
    └── Load(W32)
        └── Add
            ├── Rs1
            └── Imm  ← here
```

Parser sees `Expr::Path` with identifier `"Imm"`.

- **Rust** (`rust_backend.rs:149`): `self.operands.imm`
- **Lean** (`lean_backend.rs:285`): `imm.signExtend 64`

### Step 3: `Add(Rs1, Imm)`

```
WriteRd
└── Cast { from: W32, to: W64, sign: Signed }
    └── Load(W32)
        └── Add  ← here
            ├── Rs1  → (from step 1)
            └── Imm  → (from step 2)
```

Parser sees `Expr::Call` with function name `"Add"`, 2 args. Hits `binary_op`
helper which recurses into both args then applies the operation.

- **Rust** (`rust_backend.rs:168`): `cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm)`
- **Lean** (`lean_backend.rs:300`): `read rs1 s.reg + imm.signExtend 64`

### Step 4: `Load(W32, Add(Rs1, Imm))`

```
WriteRd
└── Cast { from: W32, to: W64, sign: Signed }
    └── Load(W32)  ← here
        └── Add  → (from step 3)
```

Parser sees `Expr::Call` with function name `"Load"`, 2 args. First arg is
width `W32`, second is the address expression (step 3).

- **Rust** (`rust_backend.rs:222`): Width `W32` → `cpu.mmu.load_word`. Returns:
  ```rust
  {
      let (val, _) = cpu.mmu.load_word(
          cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64
      ).expect("MMU load error");
      val as i64
  }
  ```

- **Lean** (`lean_backend.rs:355`): Width `W32` → `read_word`. Returns:
  `read_word (read rs1 s.reg + imm.signExtend 64) s`.
  Also: `alignment_mask("W32")` returns `Some("3")`, so pushes
  `AlignCheck { addr: "read rs1 s.reg + imm.signExtend 64", mask: "3" }`
  into `self.align_checks`.
