/// Bit width for type conversions and memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
    Xlen,
}

/// Signed or unsigned interpretation for type conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignMode {
    Signed,
    Unsigned,
}

/// An expression — pure computation, no side effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstExpr {
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
    Add(Box<AstExpr>, Box<AstExpr>),
    Sub(Box<AstExpr>, Box<AstExpr>),
    Mul(Box<AstExpr>, Box<AstExpr>),
    /// Upper bits of signed × signed multiplication.
    MulHigh(Box<AstExpr>, Box<AstExpr>),
    /// Upper bits of signed × unsigned multiplication.
    MulHighSU(Box<AstExpr>, Box<AstExpr>),
    /// Upper bits of unsigned × unsigned multiplication.
    MulHighU(Box<AstExpr>, Box<AstExpr>),
    Div(Box<AstExpr>, Box<AstExpr>),
    DivU(Box<AstExpr>, Box<AstExpr>),
    Rem(Box<AstExpr>, Box<AstExpr>),
    RemU(Box<AstExpr>, Box<AstExpr>),

    // Bitwise
    And(Box<AstExpr>, Box<AstExpr>),
    Or(Box<AstExpr>, Box<AstExpr>),
    Xor(Box<AstExpr>, Box<AstExpr>),
    Not(Box<AstExpr>),

    // Shifts
    Sll(Box<AstExpr>, Box<AstExpr>),
    Srl(Box<AstExpr>, Box<AstExpr>),
    Sra(Box<AstExpr>, Box<AstExpr>),

    // Comparison
    Eq(Box<AstExpr>, Box<AstExpr>),
    Ne(Box<AstExpr>, Box<AstExpr>),
    Lt(Box<AstExpr>, Box<AstExpr>),
    LtU(Box<AstExpr>, Box<AstExpr>),
    Ge(Box<AstExpr>, Box<AstExpr>),
    GeU(Box<AstExpr>, Box<AstExpr>),

    // Type conversion
    /// Explicit cast between widths. Sign mode determines extension behavior
    /// when `to > from`. When `to <= from` (truncation), sign is irrelevant
    /// but still required for uniformity.
    Cast { from: Width, to: Width, sign: SignMode, expr: Box<AstExpr> },

    // Unary
    TrailingZeros(Box<AstExpr>),

    // Conditional
    If(Box<AstExpr>, Box<AstExpr>, Box<AstExpr>),

    // Xlen-dependent value
    /// Select between two expressions based on xlen.
    XlenMatch { bit32: Box<AstExpr>, bit64: Box<AstExpr> },

    // Local binding
    /// Bind a named intermediate value.
    Let(String, Box<AstExpr>, Box<AstExpr>),
    /// Reference a named intermediate value.
    Var(String),
}

/// A statement — has side effects on CPU state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstStmt {
    /// Write to the instruction's destination register.
    WriteRd(AstExpr),
    /// Write to an arbitrary register by index.
    WriteReg(u8, AstExpr),
    /// Store to memory: width, address, value.
    Store(Width, AstExpr, AstExpr),
    /// Set PC to expression.
    WritePc(AstExpr),
    /// Conditional branch: if condition is true, set PC to target.
    Branch(AstExpr, AstExpr),
    /// Assert two expressions are equal (Lean: Prop, Rust: assert_eq!).
    Assert(AstExpr, AstExpr),
    /// Load from memory: bind result to named variable (width, address).
    /// Alignment is checked by backends; misaligned access sets error flag.
    Load(String, Width, AstExpr),
    /// Sequence of statements.
    Seq(Vec<AstStmt>),
    /// Bind a named value for use in subsequent statements.
    LetStmt(String, AstExpr),
    Nop,
}
