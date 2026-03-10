//! Rust backend: generates `ast_exec` method body from instruction AST.
//!
//! Walks the syn expression tree representing the DSL in `fn ast() -> Stmt`
//! and produces Rust `TokenStream2` that executes the same semantics against
//! `Cpu` and `MMU`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, ExprCall, ExprStruct};

use crate::parse::*;

/// Parse the AST expression tree and generate the corresponding Rust exec code.
pub fn generate_exec_body(block: &syn::Block) -> Result<TokenStream2, syn::Error> {
    let stmts = &block.stmts;
    if stmts.is_empty() {
        return Err(syn::Error::new_spanned(block, "ast() body is empty"));
    }

    let last = stmts.last().unwrap();
    let ret_expr = match last {
        syn::Stmt::Expr(expr, _) => expr,
        _ => return Err(syn::Error::new_spanned(last, "ast() must end with a Stmt expression")),
    };

    generate_ast_stmt(ret_expr)
}

/// Generate Rust code from a Stmt AST node.
fn generate_ast_stmt(expr: &Expr) -> Result<TokenStream2, syn::Error> {
    match expr {
        Expr::Call(call) => {
            let func_name = path_ident_name(&call.func)?;
            match func_name.as_str() {
                "WriteRd" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "WriteRd takes 1 argument"));
                    }
                    let val = generate_ast_expr(&call.args[0])?;
                    Ok(quote! {
                        let _rd_val = #val;
                        cpu.write_register(self.operands.rd as usize, _rd_val);
                    })
                }
                "Nop" => Ok(quote! {}),
                "Seq" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "Seq takes 1 argument (array)"));
                    }
                    match &call.args[0] {
                        Expr::Array(arr) => {
                            let stmts: Result<Vec<_>, _> = arr.elems.iter().map(|e| generate_ast_stmt(e)).collect();
                            let stmts = stmts?;
                            Ok(quote! { #(#stmts)* })
                        }
                        _ => Err(syn::Error::new_spanned(&call.args[0], "Seq expects an array literal")),
                    }
                }
                "Store" => {
                    if call.args.len() != 3 {
                        return Err(syn::Error::new_spanned(call, "Store takes 3 arguments: width, addr, value"));
                    }
                    let width = &call.args[0];
                    let addr = generate_ast_expr(&call.args[1])?;
                    let val = generate_ast_expr(&call.args[2])?;
                    let width_name = path_ident_name_expr(width)?;
                    let store_call = match width_name.as_str() {
                        "W8" => quote! { cpu.mmu.store(#addr as u64, #val as u8).ok().unwrap() },
                        "W16" => quote! { cpu.mmu.store_halfword(#addr as u64, #val as u16).ok().unwrap() },
                        "W32" => quote! { cpu.mmu.store_word(#addr as u64, #val as u32).ok().unwrap() },
                        "W64" => quote! { cpu.mmu.store_doubleword(#addr as u64, #val as u64).ok().unwrap() },
                        _ => return Err(syn::Error::new_spanned(width, "unsupported Store width")),
                    };
                    Ok(quote! {
                        crate::instruction::RecordWrite::record_write(_ram_access, #store_call);
                    })
                }
                "WritePc" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "WritePc takes 1 argument"));
                    }
                    let val = generate_ast_expr(&call.args[0])?;
                    Ok(quote! { cpu.pc = #val as u64; })
                }
                "Branch" => {
                    if call.args.len() != 2 {
                        return Err(syn::Error::new_spanned(call, "Branch takes 2 arguments: condition, target"));
                    }
                    let cond = generate_ast_expr(&call.args[0])?;
                    let target = generate_ast_expr(&call.args[1])?;
                    Ok(quote! {
                        if #cond != 0 {
                            cpu.pc = #target as u64;
                        }
                    })
                }
                "Assert" => {
                    if call.args.len() != 2 {
                        return Err(syn::Error::new_spanned(call, "Assert takes 2 arguments"));
                    }
                    let lhs = generate_ast_expr(&call.args[0])?;
                    let rhs = generate_ast_expr(&call.args[1])?;
                    Ok(quote! { assert_eq!(#lhs, #rhs); })
                }
                "WriteReg" => {
                    if call.args.len() != 2 {
                        return Err(syn::Error::new_spanned(call, "WriteReg takes 2 arguments: reg, value"));
                    }
                    let reg = generate_ast_expr(&call.args[0])?;
                    let val = generate_ast_expr(&call.args[1])?;
                    Ok(quote! { cpu.write_register(#reg as usize, #val); })
                }
                "LetStmt" => {
                    if call.args.len() != 2 {
                        return Err(syn::Error::new_spanned(call, "LetStmt takes 2 arguments: name, value"));
                    }
                    let name = match &call.args[0] {
                        Expr::Lit(lit) => match &lit.lit {
                            syn::Lit::Str(s) => syn::Ident::new(&s.value(), s.span()),
                            _ => return Err(syn::Error::new_spanned(&call.args[0], "LetStmt name must be a string literal")),
                        },
                        _ => return Err(syn::Error::new_spanned(&call.args[0], "LetStmt name must be a string literal")),
                    };
                    let val = generate_ast_expr(&call.args[1])?;
                    Ok(quote! { let #name = #val; })
                }
                "Load" => {
                    if call.args.len() != 3 {
                        return Err(syn::Error::new_spanned(call, "Load takes 3 arguments: name, width, addr"));
                    }
                    let name = match &call.args[0] {
                        Expr::Lit(lit) => match &lit.lit {
                            syn::Lit::Str(s) => syn::Ident::new(&s.value(), s.span()),
                            _ => return Err(syn::Error::new_spanned(&call.args[0], "Load name must be a string literal")),
                        },
                        _ => return Err(syn::Error::new_spanned(&call.args[0], "Load name must be a string literal")),
                    };
                    let width_name = path_ident_name_expr(&call.args[1])?;
                    let addr = generate_ast_expr(&call.args[2])?;
                    let load_call = match width_name.as_str() {
                        "W8" => quote! { cpu.mmu.load(#addr as u64).expect("MMU load error") },
                        "W16" => quote! { cpu.mmu.load_halfword(#addr as u64).expect("MMU load error") },
                        "W32" => quote! { cpu.mmu.load_word(#addr as u64).expect("MMU load error") },
                        "W64" => quote! { cpu.mmu.load_doubleword(#addr as u64).expect("MMU load error") },
                        _ => return Err(syn::Error::new_spanned(&call.args[1], "unsupported Load width")),
                    };
                    Ok(quote! {
                        let (#name, _ram_read) = #load_call;
                        crate::instruction::RecordRead::record_read(_ram_access, _ram_read);
                        let #name = #name as i64;
                    })
                }
                _ => Err(syn::Error::new_spanned(call, format!("unknown Stmt variant: {}", func_name))),
            }
        }
        Expr::Path(p) => {
            let name = path_ident_name_expr(expr)?;
            if name == "Nop" {
                Ok(quote! {})
            } else {
                Err(syn::Error::new_spanned(p, format!("unknown Stmt: {}", name)))
            }
        }
        _ => Err(syn::Error::new_spanned(expr, "expected a Stmt variant")),
    }
}

/// Generate a Rust expression from an Expr AST node.
fn generate_ast_expr(expr: &Expr) -> Result<TokenStream2, syn::Error> {
    match expr {
        // Terminals: Rs1, Rs2, Imm, Pc, Advice, MostNegative
        Expr::Path(p) => {
            let name = path_to_string(p)?;
            match name.as_str() {
                "Rs1" => Ok(quote! { cpu.x[self.operands.rs1 as usize] }),
                "Rs2" => Ok(quote! { cpu.x[self.operands.rs2 as usize] }),
                "Imm" => Ok(quote! { self.operands.imm }),
                "Pc" => Ok(quote! { (cpu.pc as i64) }),
                "Advice" => Ok(quote! { (self.advice as i64) }),
                "MostNegative" => Ok(quote! { cpu.most_negative() }),
                _ => Err(syn::Error::new_spanned(p, format!("unknown terminal: {}", name))),
            }
        }
        // Function-style calls: Add(a, b), Lit(42), Reg(5), etc.
        Expr::Call(call) => {
            let func_name = path_ident_name(&call.func)?;
            match func_name.as_str() {
                "Lit" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "Lit takes 1 argument"));
                    }
                    let val = &call.args[0];
                    Ok(quote! { (#val as i64) })
                }
                "Reg" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "Reg takes 1 argument"));
                    }
                    let idx = &call.args[0];
                    Ok(quote! { cpu.x[#idx as usize] })
                }
                // Binary arithmetic
                "Add" => binary_op(call, |a, b| quote! { #a.wrapping_add(#b) }),
                "Sub" => binary_op(call, |a, b| quote! { #a.wrapping_sub(#b) }),
                "Mul" => binary_op(call, |a, b| quote! { #a.wrapping_mul(#b) }),
                "Div" => binary_op(call, |a, b| quote! { #a.wrapping_div(#b) }),
                "DivU" => binary_op(call, |a, b| quote! { ((#a as u64).wrapping_div(#b as u64) as i64) }),
                "Rem" => binary_op(call, |a, b| quote! { #a.wrapping_rem(#b) }),
                "RemU" => binary_op(call, |a, b| quote! { ((#a as u64).wrapping_rem(#b as u64) as i64) }),

                // Bitwise
                "And" => binary_op(call, |a, b| quote! { (#a & #b) }),
                "Or" => binary_op(call, |a, b| quote! { (#a | #b) }),
                "Xor" => binary_op(call, |a, b| quote! { (#a ^ #b) }),
                "Not" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "Not takes 1 argument"));
                    }
                    let a = generate_ast_expr(&call.args[0])?;
                    Ok(quote! { (!#a) })
                }

                // Shifts
                "Sll" => binary_op(call, |a, b| quote! { #a.wrapping_shl(#b as u32) }),
                "Srl" => binary_op(call, |a, b| quote! { ((#a as u64).wrapping_shr(#b as u32) as i64) }),
                "Sra" => binary_op(call, |a, b| quote! { #a.wrapping_shr(#b as u32) }),

                // Comparisons (return 1 or 0)
                "Eq" => binary_op(call, |a, b| quote! { if #a == #b { 1i64 } else { 0i64 } }),
                "Ne" => binary_op(call, |a, b| quote! { if #a != #b { 1i64 } else { 0i64 } }),
                "Lt" => binary_op(call, |a, b| quote! { if #a < #b { 1i64 } else { 0i64 } }),
                "LtU" => binary_op(call, |a, b| quote! { if (#a as u64) < (#b as u64) { 1i64 } else { 0i64 } }),
                "Ge" => binary_op(call, |a, b| quote! { if #a >= #b { 1i64 } else { 0i64 } }),
                "GeU" => binary_op(call, |a, b| quote! { if (#a as u64) >= (#b as u64) { 1i64 } else { 0i64 } }),

                // MulHigh variants
                "MulHigh" => binary_op(call, |a, b| quote! {
                    ((((#a as i128).wrapping_mul(#b as i128)) >> 64) as i64)
                }),
                "MulHighSU" => binary_op(call, |a, b| quote! {
                    ((((#a as i128).wrapping_mul(#b as u64 as i128)) >> 64) as i64)
                }),
                "MulHighU" => binary_op(call, |a, b| quote! {
                    ((((#a as u64 as u128).wrapping_mul(#b as u64 as u128)) >> 64) as i64)
                }),

                // TrailingZeros(expr)
                "TrailingZeros" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "TrailingZeros takes 1 argument"));
                    }
                    let inner = generate_ast_expr(&call.args[0])?;
                    Ok(quote! { ((#inner as u64).trailing_zeros() as i64) })
                }

                // If(cond, then, else)
                "If" => {
                    if call.args.len() != 3 {
                        return Err(syn::Error::new_spanned(call, "If takes 3 arguments"));
                    }
                    let cond = generate_ast_expr(&call.args[0])?;
                    let then_expr = generate_ast_expr(&call.args[1])?;
                    let else_expr = generate_ast_expr(&call.args[2])?;
                    Ok(quote! { if #cond != 0 { #then_expr } else { #else_expr } })
                }

                // Let(name, value, body)
                "Let" => {
                    if call.args.len() != 3 {
                        return Err(syn::Error::new_spanned(call, "Let takes 3 arguments: name, value, body"));
                    }
                    let name = match &call.args[0] {
                        Expr::Lit(lit) => match &lit.lit {
                            syn::Lit::Str(s) => syn::Ident::new(&s.value(), s.span()),
                            _ => return Err(syn::Error::new_spanned(&call.args[0], "Let name must be a string literal")),
                        },
                        _ => return Err(syn::Error::new_spanned(&call.args[0], "Let name must be a string literal")),
                    };
                    let val = generate_ast_expr(&call.args[1])?;
                    let body = generate_ast_expr(&call.args[2])?;
                    Ok(quote! { { let #name = #val; #body } })
                }

                // Var(name)
                "Var" => {
                    if call.args.len() != 1 {
                        return Err(syn::Error::new_spanned(call, "Var takes 1 argument"));
                    }
                    let name = match &call.args[0] {
                        Expr::Lit(lit) => match &lit.lit {
                            syn::Lit::Str(s) => syn::Ident::new(&s.value(), s.span()),
                            _ => return Err(syn::Error::new_spanned(&call.args[0], "Var name must be a string literal")),
                        },
                        _ => return Err(syn::Error::new_spanned(&call.args[0], "Var name must be a string literal")),
                    };
                    Ok(quote! { #name })
                }

                _ => Err(syn::Error::new_spanned(call, format!("unknown Expr variant: {}", func_name))),
            }
        }
        // Struct-style: Cast { from: W32, to: W64, sign: Signed, expr: ... }
        Expr::Struct(s) => {
            let name = path_to_string_from_path(&s.path)?;
            match name.as_str() {
                "Cast" => generate_ast_cast(s),
                "XlenMatch" => {
                    let bit32 = find_struct_field(s, "bit32")?;
                    let bit64 = find_struct_field(s, "bit64")?;
                    let gen32 = generate_ast_expr(bit32)?;
                    let gen64 = generate_ast_expr(bit64)?;
                    Ok(quote! {
                        match cpu.xlen {
                            crate::emulator::cpu::Xlen::Bit32 => #gen32,
                            crate::emulator::cpu::Xlen::Bit64 => #gen64,
                        }
                    })
                }
                _ => Err(syn::Error::new_spanned(s, format!("unknown struct Expr: {}", name))),
            }
        }
        // Negation: -1, -8, etc.
        Expr::Unary(u) => {
            Ok(quote! { (#u as i64) })
        }
        // Integer literal
        Expr::Lit(lit) => {
            Ok(quote! { (#lit as i64) })
        }
        _ => Err(syn::Error::new_spanned(expr, "unsupported expression form")),
    }
}

/// Generate code for Cast { from, to, sign, expr }.
fn generate_ast_cast(s: &ExprStruct) -> Result<TokenStream2, syn::Error> {
    let from = find_struct_field(s, "from")?;
    let to = find_struct_field(s, "to")?;
    let sign = find_struct_field(s, "sign")?;
    let inner = find_struct_field(s, "expr")?;

    let from_name = path_ident_name_expr(from)?;
    let to_name = path_ident_name_expr(to)?;
    let sign_name = path_ident_name_expr(sign)?;
    let inner_gen = generate_ast_expr(inner)?;

    let signed = match sign_name.as_str() {
        "Signed" => true,
        "Unsigned" => false,
        _ => return Err(syn::Error::new_spanned(sign, "sign must be Signed or Unsigned")),
    };

    match (from_name.as_str(), to_name.as_str()) {
        // Truncation to fixed widths
        (_, "W8") => Ok(quote! { ((#inner_gen as u8) as i64) }),
        (_, "W16") => Ok(quote! { ((#inner_gen as u16) as i64) }),
        (_, "W32") if !signed => Ok(quote! { ((#inner_gen as u32) as i64) }),
        (_, "W32") if signed => Ok(quote! { (#inner_gen as i32 as i64) }),
        // To W64
        ("W8", "W64") if signed => Ok(quote! { (#inner_gen as i8 as i64) }),
        ("W8", "W64") => Ok(quote! { ((#inner_gen as u8) as i64) }),
        ("W16", "W64") if signed => Ok(quote! { (#inner_gen as i16 as i64) }),
        ("W16", "W64") => Ok(quote! { ((#inner_gen as u16) as i64) }),
        ("W32", "W64") if signed => Ok(quote! { (#inner_gen as i32 as i64) }),
        ("W32", "W64") => Ok(quote! { ((#inner_gen as u32) as i64) }),
        ("Xlen", "W64") if signed => Ok(quote! { cpu.sign_extend(#inner_gen) }),
        ("Xlen", "W64") => Ok(quote! { (cpu.unsigned_data(#inner_gen) as i64) }),
        ("W64", "W64") => Ok(quote! { (#inner_gen) }),
        // To Xlen
        ("W8", "Xlen") if signed => Ok(quote! { cpu.sign_extend(#inner_gen as i8 as i64) }),
        ("W8", "Xlen") => Ok(quote! { ((#inner_gen as u8) as i64) }),
        ("W16", "Xlen") if signed => Ok(quote! { cpu.sign_extend(#inner_gen as i16 as i64) }),
        ("W16", "Xlen") => Ok(quote! { ((#inner_gen as u16) as i64) }),
        ("W32", "Xlen") if signed => Ok(quote! { cpu.sign_extend(#inner_gen as i32 as i64) }),
        ("W32", "Xlen") => Ok(quote! { ((#inner_gen as u32) as i64) }),
        ("W64", "Xlen") => Ok(quote! { (#inner_gen) }),
        ("Xlen", "Xlen") if signed => Ok(quote! { cpu.sign_extend(#inner_gen) }),
        ("Xlen", "Xlen") => Ok(quote! { (cpu.unsigned_data(#inner_gen) as i64) }),
        _ => Err(syn::Error::new_spanned(s, format!("unsupported Cast: {} -> {}", from_name, to_name))),
    }
}

/// Generate code for a binary operation.
fn binary_op(
    call: &ExprCall,
    gen: impl FnOnce(TokenStream2, TokenStream2) -> TokenStream2,
) -> Result<TokenStream2, syn::Error> {
    if call.args.len() != 2 {
        return Err(syn::Error::new_spanned(call, "binary op takes 2 arguments"));
    }
    let a = generate_ast_expr(&call.args[0])?;
    let b = generate_ast_expr(&call.args[1])?;
    Ok(gen(a, b))
}
