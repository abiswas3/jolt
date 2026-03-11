#[path = "../../../tracer-macros/src/parse.rs"]
mod parse;
#[path = "../../../tracer-macros/src/lean_backend.rs"]
mod lean_backend;

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gen-lean <instruction_module> [name_override]");
        std::process::exit(1);
    }

    let instr = &args[1];
    let name_override = args.get(2).map(|s| s.as_str());

    // Read the source file
    let jolt_root = env!("CARGO_MANIFEST_DIR").to_string() + "/../..";
    let src_path = format!("{}/tracer/src/instruction/{}.rs", jolt_root, instr);
    let source = fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", src_path, e));

    // Extract the ast() method body by finding the impl block with #[tracer_macros::gen_exec]
    let file = syn::parse_file(&source).expect("Failed to parse source file");

    for item in &file.items {
        if let syn::Item::Impl(impl_block) = item {
            // Check if this impl has the gen_exec attribute
            let has_gen_exec = impl_block.attrs.iter().any(|attr| {
                attr.path().segments.iter().any(|seg| seg.ident == "gen_exec")
            });
            if !has_gen_exec {
                continue;
            }

            // Find the ast() method
            for impl_item in &impl_block.items {
                if let syn::ImplItem::Fn(method) = impl_item {
                    if method.sig.ident == "ast" {
                        let lean_name = name_override.unwrap_or(instr);
                        match lean_backend::generate_lean(lean_name, &method.block) {
                            Ok(lean_code) => {
                                println!("{}", lean_code);
                                return;
                            }
                            Err(e) => {
                                eprintln!("Error generating Lean: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("No fn ast() found with #[tracer_macros::gen_exec] in {}", src_path);
    std::process::exit(1);
}
