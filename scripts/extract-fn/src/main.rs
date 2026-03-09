use std::io::Read;
use syn::{File, ImplItem, Item};

fn extract_from_items(items: &[Item], names: &[String]) {
    for item in items {
        match item {
            Item::Impl(impl_block) => {
                for impl_item in &impl_block.items {
                    if let ImplItem::Fn(method) = impl_item {
                        let fn_name = method.sig.ident.to_string();
                        if names.contains(&fn_name) {
                            println!("=== {} ===", fn_name);
                            let wrapped = format!(
                                "impl Dummy {{ {} }}",
                                quote::quote! { #method }
                            );
                            let parsed = syn::parse_file(&wrapped).unwrap();
                            let formatted = prettyplease::unparse(&parsed);
                            for line in formatted.lines() {
                                let trimmed = line.trim();
                                if trimmed == "impl Dummy {" || trimmed == "}" {
                                    continue;
                                }
                                println!("{}", line);
                            }
                            println!();
                        }
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    extract_from_items(items, names);
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let names: Vec<String> = std::env::args().skip(1).collect();
    if names.is_empty() {
        eprintln!("Usage: extract-fn <fn_name> [fn_name...]");
        std::process::exit(1);
    }

    let mut src = String::new();
    std::io::stdin().read_to_string(&mut src).expect("failed to read stdin");

    let file: File = syn::parse_file(&src).expect("failed to parse");
    extract_from_items(&file.items, &names);
}
