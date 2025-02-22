/*
Corresponds to wgpu/compiler/shader.rs. The compiled executable kernel functions, stored
as dynamically linked libraries.
*/

use alloc::fmt::Display;

use cubecl_core::compute::Binding;

#[derive(Debug, Clone)]
pub struct FfiFunction {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub kernel_name: String,
}

impl Display for FfiFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FfiFunction {{ kernel_name: {} }}", self.kernel_name)
    }
}
