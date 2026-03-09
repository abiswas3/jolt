mod parse;
mod rust_backend;
#[allow(dead_code)]
mod lean_backend;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemImpl, ImplItem, ImplItemFn};

/// Attribute macro for instruction impl blocks.
///
/// Reads the `fn ast() -> Stmt` method, and generates a corresponding
/// `fn ast_exec(&self, cpu: &mut Cpu, ram_access: &mut ...)` method
/// that executes the same semantics described by the AST.
#[proc_macro_attribute]
pub fn gen_exec(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);

    // Find the ast() method
    let ast_fn = impl_block.items.iter().find_map(|item| {
        if let ImplItem::Fn(method) = item {
            if method.sig.ident == "ast" {
                return Some(method.clone());
            }
        }
        None
    });

    let ast_fn = match ast_fn {
        Some(f) => f,
        None => {
            return syn::Error::new_spanned(&impl_block, "gen_exec requires fn ast() -> Stmt")
                .to_compile_error()
                .into();
        }
    };

    // Extract the return expression from the ast() body.
    let body = &ast_fn.block;
    let exec_body = match rust_backend::generate_exec_body(body) {
        Ok(tokens) => tokens,
        Err(e) => return e.to_compile_error().into(),
    };

    // Get the type name from the impl block
    let self_ty = &impl_block.self_ty;

    // Generate ast_exec method
    let ast_exec_method: ImplItemFn = syn::parse_quote! {
        fn ast_exec(&self, cpu: &mut crate::emulator::cpu::Cpu, ram_access: &mut <#self_ty as crate::instruction::RISCVInstruction>::RAMAccess) {
            let mut _ram_read_out: Option<crate::instruction::RAMRead> = None;
            let mut _ram_write_out: Option<crate::instruction::RAMWrite> = None;
            #exec_body
            crate::instruction::assign_ram_access(ram_access, _ram_read_out, _ram_write_out);
        }
    };

    // Remove the ast() method from the output — it uses DSL syntax
    // that isn't valid Rust. Only the generated ast_exec() remains.
    impl_block.items.retain(|item| {
        if let ImplItem::Fn(method) = item {
            method.sig.ident != "ast"
        } else {
            true
        }
    });

    impl_block.items.push(ImplItem::Fn(ast_exec_method));

    TokenStream::from(quote! { #impl_block })
}

// Lean backend is compiled as part of this crate but invoked via
// a separate binary (gen-lean) that depends on tracer-macros internals.
// proc-macro crates cannot re-export non-macro items, so the lean_backend
// module is accessed directly by the binary crate.
