/// Bit width for type conversions and memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
    Xlen,
}

/// An expression — pure computation, no side effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    // Terminals
    Rs1,
    Rs2,
    Imm,
    Pc,
    Reg(u8),
    Lit(i128),
    /// Externally-provided value (Lean: free variable, Rust: self.advice).
    Advice,
    /// Xlen-dependent most negative value (i32::MIN or i64::MIN).
    MostNegative,

    // Arithmetic
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    /// Upper bits of signed × signed multiplication.
    MulHigh(Box<Expr>, Box<Expr>),
    /// Upper bits of signed × unsigned multiplication.
    MulHighSU(Box<Expr>, Box<Expr>),
    /// Upper bits of unsigned × unsigned multiplication.
    MulHighU(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    DivU(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),
    RemU(Box<Expr>, Box<Expr>),

    // Bitwise
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),

    // Shifts
    Sll(Box<Expr>, Box<Expr>),
    Srl(Box<Expr>, Box<Expr>),
    Sra(Box<Expr>, Box<Expr>),

    // Comparison
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    LtU(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    GeU(Box<Expr>, Box<Expr>),

    // Type conversion
    /// Sign-extend from `from` bits to `to` bits.
    Sext { from: Width, to: Width, expr: Box<Expr> },
    /// Zero-extend from `from` bits to `to` bits.
    Zext { from: Width, to: Width, expr: Box<Expr> },
    /// Truncate to `to` bits.
    Trunc(Width, Box<Expr>),

    // Memory
    Load(Width, Box<Expr>),

    // Unary
    TrailingZeros(Box<Expr>),

    // Conditional
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    // Xlen-dependent value
    /// Select between two expressions based on xlen.
    XlenMatch { bit32: Box<Expr>, bit64: Box<Expr> },

    // Local binding
    /// Bind a named intermediate value.
    Let(String, Box<Expr>, Box<Expr>),
    /// Reference a named intermediate value.
    Var(String),
}

/// A statement — has side effects on CPU state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// Write to the instruction's destination register.
    WriteRd(Expr),
    /// Write to an arbitrary register by index.
    WriteReg(u8, Expr),
    /// Store to memory: width, address, value.
    Store(Width, Expr, Expr),
    /// Set PC to expression.
    WritePc(Expr),
    /// Conditional branch: if condition is true, set PC to target.
    Branch(Expr, Expr),
    /// Assert two expressions are equal (Lean: Prop, Rust: assert_eq!).
    Assert(Expr, Expr),
    /// Sequence of statements.
    Seq(Vec<Stmt>),
    /// Bind a named value for use in subsequent statements.
    LetStmt(String, Expr),
    Nop,
}
