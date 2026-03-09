//! Shared helpers for parsing syn AST nodes from instruction `fn ast()` methods.

use syn::{Expr, ExprPath, ExprStruct};

/// Extract the last identifier from a path used as a function name.
pub fn path_ident_name(func: &Expr) -> Result<String, syn::Error> {
    match func {
        Expr::Path(p) => path_to_string(p),
        _ => Err(syn::Error::new_spanned(func, "expected path")),
    }
}

pub fn path_ident_name_expr(expr: &Expr) -> Result<String, syn::Error> {
    match expr {
        Expr::Path(p) => path_to_string(p),
        _ => Err(syn::Error::new_spanned(expr, "expected path identifier")),
    }
}

pub fn path_to_string(p: &ExprPath) -> Result<String, syn::Error> {
    path_to_string_from_path(&p.path)
}

pub fn path_to_string_from_path(p: &syn::Path) -> Result<String, syn::Error> {
    p.segments
        .last()
        .map(|s| s.ident.to_string())
        .ok_or_else(|| syn::Error::new_spanned(p, "empty path"))
}

/// Find a named field in a struct expression.
pub fn find_struct_field<'a>(s: &'a ExprStruct, name: &str) -> Result<&'a Expr, syn::Error> {
    for field in &s.fields {
        if let syn::Member::Named(ident) = &field.member {
            if ident == name {
                return Ok(&field.expr);
            }
        }
    }
    Err(syn::Error::new_spanned(s, format!("missing field: {}", name)))
}
