use alloc::fmt::Debug;
use cranelift::prelude::FunctionBuilder;
use cranelift_codegen::ir::Function;
use cubecl_core::{Compiler, ExecutionMode};

use super::FfiFunction;

#[derive(Clone)]
pub struct FunctionCompiler {
    //builder: FunctionBuilder<'static>,
    exec_mode: ExecutionMode,
}

impl Compiler for FunctionCompiler {
    type Representation = FfiFunction;

    type CompilationOptions = ();

    fn compile(
        &mut self,
        kernel: cubecl_core::prelude::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        todo!()
    }

    fn elem_size(&self, elem: cubecl_core::ir::Elem) -> usize {
        todo!()
    }
}

impl Debug for FunctionCompiler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCompiler")
            .field("exec_mode", &self.exec_mode)
            //.field("builder_func", &self.builder.func)
            .finish()
    }
}
