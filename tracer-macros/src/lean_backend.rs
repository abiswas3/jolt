//! Lean backend: generates Lean 4 definitions from instruction AST.
//!
//! Produces pure-functional Lean code operating on `State` with:
//! - `s.reg` : RegFile (BitVec 5 → BitVec 64)
//! - `s.mem` : Memory  (BitVec 64 → BitVec 8)
//! - `s.pc`  : BitVec 64
//! - `s.error`: Bool
//!
//! Memory operations include explicit alignment preconditions.

use syn::{Expr, ExprStruct};

use crate::parse::*;

/// Width tag → Lean BitVec width in bits.
fn width_bits(name: &str) -> Result<&'static str, String> {
    match name {
        "W8" => Ok("8"),
        "W16" => Ok("16"),
        "W32" => Ok("32"),
        "W64" => Ok("64"),
        _ => Err(format!("unsupported width: {}", name)),
    }
}

/// Width tag → alignment mask (bytes - 1). W8 has no alignment requirement.
fn alignment_mask(name: &str) -> Option<&'static str> {
    match name {
        "W16" => Some("1"),
        "W32" => Some("3"),
        "W64" => Some("7"),
        _ => None,
    }
}

/// Width tag → Lean memory helper name.
fn load_fn(name: &str) -> Result<&'static str, String> {
    match name {
        "W8" => Ok("read_byte"),
        "W16" => Ok("read_halfword"),
        "W32" => Ok("read_word"),
        "W64" => Ok("read_doubleword"),
        _ => Err(format!("unsupported Load width: {}", name)),
    }
}

fn store_fn(name: &str) -> Result<&'static str, String> {
    match name {
        "W8" => Ok("store_byte"),
        "W16" => Ok("store_halfword"),
        "W32" => Ok("store_word"),
        "W64" => Ok("store_doubleword"),
        _ => Err(format!("unsupported Store width: {}", name)),
    }
}

/// Generate a complete Lean definition for an instruction.
///
/// `name` is the instruction name (e.g. "lw", "amoaddw").
/// `block` is the parsed body of `fn ast() -> Stmt`.
///
/// Returns a String of Lean 4 code.
pub fn generate_lean(name: &str, block: &syn::Block) -> Result<String, syn::Error> {
    let stmts = &block.stmts;
    if stmts.is_empty() {
        return Err(syn::Error::new_spanned(block, "ast() body is empty"));
    }

    let last = stmts.last().unwrap();
    let ret_expr = match last {
        syn::Stmt::Expr(expr, _) => expr,
        _ => return Err(syn::Error::new_spanned(last, "ast() must end with a Stmt expression")),
    };

    let mut ctx = LeanCtx { indent: 1 };
    let body = ctx.generate_ast_stmt(ret_expr)?;

    Ok(format!(
        "def {} (rs1 rs2 rd : BitVec 5) (imm : BitVec 12) (s : State) : State :=\n{}",
        name, body
    ))
}

struct LeanCtx {
    indent: usize,
}

impl LeanCtx {
    fn pad(&self) -> String {
        "  ".repeat(self.indent)
    }

    fn generate_ast_stmt(&mut self, expr: &Expr) -> Result<String, syn::Error> {
        match expr {
            Expr::Call(call) => {
                let func_name = path_ident_name(&call.func)?;
                match func_name.as_str() {
                    "WriteRd" => {
                        if call.args.len() != 1 {
                            return Err(syn::Error::new_spanned(call, "WriteRd takes 1 argument"));
                        }
                        let val = self.generate_ast_expr(&call.args[0])?;
                        let pad = self.pad();
                        Ok(format!("{pad}let new_reg := write rd ({val}) s.reg\n{pad}{{ s with reg := new_reg }}"))
                    }
                    "Nop" => {
                        let pad = self.pad();
                        Ok(format!("{pad}s"))
                    }
                    "Seq" => {
                        if call.args.len() != 1 {
                            return Err(syn::Error::new_spanned(call, "Seq takes 1 argument"));
                        }
                        match &call.args[0] {
                            Expr::Array(arr) => {
                                // Thread state through: each stmt takes s and produces s
                                // We generate let-chain style
                                let items: Vec<_> = arr.elems.iter().collect();
                                self.generate_ast_seq(&items)
                            }
                            _ => Err(syn::Error::new_spanned(&call.args[0], "Seq expects an array literal")),
                        }
                    }
                    "Store" => {
                        if call.args.len() != 3 {
                            return Err(syn::Error::new_spanned(call, "Store takes 3 arguments"));
                        }
                        let width_name = path_ident_name_expr(&call.args[0])?;
                        let addr = self.generate_ast_expr(&call.args[1])?;
                        let val = self.generate_ast_expr(&call.args[2])?;
                        let pad = self.pad();
                        let sfn = store_fn(&width_name)
                            .map_err(|e| syn::Error::new_spanned(&call.args[0], e))?;
                        // Alignment check
                        let align = if let Some(mask) = alignment_mask(&width_name) {
                            format!("{pad}if ({addr}) &&& {mask}#64 ≠ 0#64 then {{ s with error := true }}\n{pad}else\n")
                        } else {
                            String::new()
                        };
                        let bits = width_bits(&width_name)
                            .map_err(|e| syn::Error::new_spanned(&call.args[0], e))?;
                        Ok(format!(
                            "{align}{pad}let new_mem := {sfn} ({addr}) (({val}).truncate {bits}) s.mem\n{pad}{{ s with mem := new_mem }}"
                        ))
                    }
                    "WritePc" => {
                        if call.args.len() != 1 {
                            return Err(syn::Error::new_spanned(call, "WritePc takes 1 argument"));
                        }
                        let val = self.generate_ast_expr(&call.args[0])?;
                        let pad = self.pad();
                        Ok(format!("{pad}{{ s with pc := ({val}).truncate 64 }}"))
                    }
                    "Branch" => {
                        if call.args.len() != 2 {
                            return Err(syn::Error::new_spanned(call, "Branch takes 2 arguments"));
                        }
                        let cond = self.generate_ast_expr(&call.args[0])?;
                        let target = self.generate_ast_expr(&call.args[1])?;
                        let pad = self.pad();
                        Ok(format!(
                            "{pad}if {cond} ≠ 0#64 then {{ s with pc := ({target}).truncate 64 }}\n{pad}else s"
                        ))
                    }
                    "LetStmt" => {
                        if call.args.len() != 2 {
                            return Err(syn::Error::new_spanned(call, "LetStmt takes 2 arguments"));
                        }
                        let name = extract_string_lit(&call.args[0])?;
                        let val = self.generate_ast_expr(&call.args[1])?;
                        let pad = self.pad();
                        Ok(format!("{pad}let {name} := {val}"))
                    }
                    "Load" => {
                        if call.args.len() != 3 {
                            return Err(syn::Error::new_spanned(call, "Load takes 3 arguments: name, width, addr"));
                        }
                        let name = extract_string_lit(&call.args[0])?;
                        let width_name = path_ident_name_expr(&call.args[1])?;
                        let addr = self.generate_ast_expr(&call.args[2])?;
                        let lfn = load_fn(&width_name)
                            .map_err(|e| syn::Error::new_spanned(&call.args[1], e))?;
                        let pad = self.pad();
                        // Alignment check inline, same as Store
                        let align = if let Some(mask) = alignment_mask(&width_name) {
                            format!("{pad}if ({addr}) &&& {mask}#64 ≠ 0#64 then {{ s with error := true }}\n{pad}else\n")
                        } else {
                            String::new()
                        };
                        Ok(format!("{align}{pad}let {name} := {lfn} ({addr}) s"))
                    }
                    "WriteReg" => {
                        if call.args.len() != 2 {
                            return Err(syn::Error::new_spanned(call, "WriteReg takes 2 arguments"));
                        }
                        let reg = self.generate_ast_expr(&call.args[0])?;
                        let val = self.generate_ast_expr(&call.args[1])?;
                        let pad = self.pad();
                        Ok(format!("{pad}let new_reg := write ({reg}).truncate s.reg ({val})\n{pad}{{ s with reg := new_reg }}"))
                    }
                    _ => Err(syn::Error::new_spanned(call, format!("unknown Stmt: {}", func_name))),
                }
            }
            Expr::Path(_) => {
                let name = path_ident_name_expr(expr)?;
                if name == "Nop" {
                    let pad = self.pad();
                    Ok(format!("{pad}s"))
                } else {
                    Err(syn::Error::new_spanned(expr, format!("unknown Stmt: {}", name)))
                }
            }
            _ => Err(syn::Error::new_spanned(expr, "expected Stmt variant")),
        }
    }

    /// Generate a sequence of statements threaded through state.
    /// The last statement is the "return", earlier ones are let-bindings on state.
    fn generate_ast_seq(&mut self, items: &[&Expr]) -> Result<String, syn::Error> {
        if items.is_empty() {
            let pad = self.pad();
            return Ok(format!("{pad}s"));
        }

        let mut lines = Vec::new();
        let pad = self.pad();

        for (i, item) in items.iter().enumerate() {
            let is_last = i == items.len() - 1;

            // Check if this is a LetStmt or Load — these just bind a value, don't produce new state
            if let Expr::Call(call) = item {
                if let Ok(name) = path_ident_name(&call.func) {
                    if name == "LetStmt" || name == "Load" {
                        let stmt = self.generate_ast_stmt(item)?;
                        lines.push(stmt);
                        continue;
                    }
                }
            }

            if is_last {
                // Last statement produces the final state
                let stmt = self.generate_ast_stmt(item)?;
                lines.push(stmt);
            } else {
                // Intermediate state-modifying statement: bind result to s
                let stmt = self.generate_ast_stmt(item)?;
                lines.push(format!("{pad}let s :="));
                self.indent += 1;
                lines.push(self.reindent_stmt(&stmt));
                self.indent -= 1;
            }
        }

        Ok(lines.join("\n"))
    }

    fn reindent_stmt(&self, _stmt: &str) -> String {
        // Stmt is already indented from generate_ast_stmt; just return as-is
        // since generate_ast_stmt uses self.pad() at current indent level
        _stmt.to_string()
    }

    fn generate_ast_expr(&self, expr: &Expr) -> Result<String, syn::Error> {
        match expr {
            Expr::Path(p) => {
                let name = path_to_string(p)?;
                match name.as_str() {
                    "Rs1" => Ok("read rs1 s.reg".to_string()),
                    "Rs2" => Ok("read rs2 s.reg".to_string()),
                    "Imm" => Ok("imm.signExtend 64".to_string()),
                    "Pc" => Ok("s.pc".to_string()),
                    _ => Err(syn::Error::new_spanned(p, format!("unknown terminal: {}", name))),
                }
            }
            Expr::Call(call) => {
                let func_name = path_ident_name(&call.func)?;
                match func_name.as_str() {
                    "Lit" => {
                        if call.args.len() != 1 {
                            return Err(syn::Error::new_spanned(call, "Lit takes 1 argument"));
                        }
                        let val = &call.args[0];
                        Ok(format!("{}#64", quote_lit(val)))
                    }
                    "Add" => self.lean_binop(&call.args, "+"),
                    "Sub" => self.lean_binop(&call.args, "-"),
                    "Mul" => self.lean_binop(&call.args, "*"),
                    "And" => self.lean_binop(&call.args, "&&&"),
                    "Or" => self.lean_binop(&call.args, "|||"),
                    "Xor" => self.lean_binop(&call.args, "^^^"),
                    "Sll" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("{a} <<< {b}"))
                    }
                    "Srl" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("{a} >>> {b}"))
                    }
                    "Sra" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("BitVec.sshiftRight {a} {b}.toNat"))
                    }
                    "Eq" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if {a} = {b} then 1#64 else 0#64"))
                    }
                    "Ne" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if {a} ≠ {b} then 1#64 else 0#64"))
                    }
                    "Lt" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if BitVec.slt {a} {b} then 1#64 else 0#64"))
                    }
                    "LtU" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if BitVec.ult {a} {b} then 1#64 else 0#64"))
                    }
                    "Ge" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if ¬ BitVec.slt {a} {b} then 1#64 else 0#64"))
                    }
                    "GeU" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        let b = self.generate_ast_expr(&call.args[1])?;
                        Ok(format!("if ¬ BitVec.ult {a} {b} then 1#64 else 0#64"))
                    }
                    "Not" => {
                        let a = self.generate_ast_expr(&call.args[0])?;
                        Ok(format!("~~~{a}"))
                    }
                    "If" => {
                        if call.args.len() != 3 {
                            return Err(syn::Error::new_spanned(call, "If takes 3 arguments"));
                        }
                        let cond = self.generate_ast_expr(&call.args[0])?;
                        let then_e = self.generate_ast_expr(&call.args[1])?;
                        let else_e = self.generate_ast_expr(&call.args[2])?;
                        Ok(format!("if {cond} ≠ 0#64 then {then_e} else {else_e}"))
                    }
                    "Var" => {
                        let name = extract_string_lit(&call.args[0])?;
                        Ok(name)
                    }
                    _ => Err(syn::Error::new_spanned(call, format!("unknown Expr: {}", func_name))),
                }
            }
            Expr::Struct(s) => {
                let name = path_to_string_from_path(&s.path)?;
                match name.as_str() {
                    "Cast" => self.generate_ast_cast(s),
                    _ => Err(syn::Error::new_spanned(s, format!("unknown struct Expr: {}", name))),
                }
            }
            Expr::Unary(_) | Expr::Lit(_) => {
                Ok(format!("{}#64", quote::quote! { #expr }))
            }
            _ => Err(syn::Error::new_spanned(expr, "unsupported expression")),
        }
    }

    fn lean_binop(&self, args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>, op: &str) -> Result<String, syn::Error> {
        if args.len() != 2 {
            return Err(syn::Error::new_spanned(&args[0], "binary op takes 2 arguments"));
        }
        let a = self.generate_ast_expr(&args[0])?;
        let b = self.generate_ast_expr(&args[1])?;
        Ok(format!("{a} {op} {b}"))
    }

    fn generate_ast_cast(&self, s: &ExprStruct) -> Result<String, syn::Error> {
        let from = find_struct_field(s, "from")?;
        let to = find_struct_field(s, "to")?;
        let sign = find_struct_field(s, "sign")?;
        let inner = find_struct_field(s, "expr")?;

        let from_name = path_ident_name_expr(from)?;
        let to_name = path_ident_name_expr(to)?;
        let sign_name = path_ident_name_expr(sign)?;
        let inner_gen = self.generate_ast_expr(inner)?;

        let signed = sign_name == "Signed";

        let from_bits = width_bits(&from_name)
            .map_err(|e| syn::Error::new_spanned(from, e))?;
        let to_bits = width_bits(&to_name)
            .map_err(|e| syn::Error::new_spanned(to, e))?;

        let from_n: u32 = from_bits.parse().unwrap();
        let to_n: u32 = to_bits.parse().unwrap();

        if to_n > from_n {
            // Widening
            if signed {
                Ok(format!("(({inner_gen}).truncate {from_bits}).signExtend {to_bits}"))
            } else {
                Ok(format!("(({inner_gen}).truncate {from_bits}).zeroExtend {to_bits}"))
            }
        } else if to_n < from_n {
            // Narrowing — truncation
            Ok(format!("({inner_gen}).truncate {to_bits}"))
        } else {
            // Same width — identity
            Ok(format!("({inner_gen})"))
        }
    }
}

fn extract_string_lit(expr: &Expr) -> Result<String, syn::Error> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Str(s) => Ok(s.value()),
            _ => Err(syn::Error::new_spanned(expr, "expected string literal")),
        },
        _ => Err(syn::Error::new_spanned(expr, "expected string literal")),
    }
}

fn quote_lit(expr: &Expr) -> String {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(i) => i.base10_digits().to_string(),
            _ => format!("{}", quote::quote! { #expr }),
        },
        Expr::Unary(_) => format!("{}", quote::quote! { #expr }),
        _ => format!("{}", quote::quote! { #expr }),
    }
}
