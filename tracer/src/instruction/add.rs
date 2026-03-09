use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = ADD,
    mask   = 0xfe00707f,
    match  = 0x00000033,
    format = FormatR,
    ram    = ()
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADD as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize].wrapping_add(cpu.x[self.operands.rs2 as usize]),
            ),
        );
    }
}

#[tracer_macros::gen_exec]
impl ADD {
    fn ast() -> Stmt {
        WriteRd(Sext { from: Xlen, to: W64, expr: Add(Rs1, Rs2) })
    }
}

impl RISCVTrace for ADD {}
