//! Kernel source locations as LLVM debug data.
//!
//! `pliron-llvm` does not convert the `Location` of an op. [`encode_locations`] puts the index of
//! each location in the metadata `cubecl.loc`, which the conversion keeps.
//! [`attach_debug_info`] reads it back from the parsed LLVM module, and replaces it with a `!dbg`
//! location: one `DISubprogram` for each function, and one for each inlined `#[cube]` function.

use crate::{prelude::*, shared::llvm_module::LlvmModule};
use cubecl_core::ir::settings::DebugInfo;
use llvm_sys::{
    LLVMModuleFlagBehavior,
    core::{
        LLVMAddModuleFlag, LLVMConstInt, LLVMGetFirstBasicBlock, LLVMGetFirstFunction,
        LLVMGetFirstInstruction, LLVMGetMDKindIDInContext, LLVMGetMDNodeNumOperands,
        LLVMGetMDNodeOperands, LLVMGetMDString, LLVMGetMetadata, LLVMGetNextBasicBlock,
        LLVMGetNextFunction, LLVMGetNextInstruction, LLVMGetValueName2, LLVMInt32TypeInContext,
        LLVMSetMetadata, LLVMValueAsMetadata,
    },
    debuginfo::{
        LLVMCreateDIBuilder, LLVMDIBuilderCreateCompileUnit, LLVMDIBuilderCreateDebugLocation,
        LLVMDIBuilderCreateFile, LLVMDIBuilderCreateFunction, LLVMDIBuilderCreateSubroutineType,
        LLVMDIBuilderFinalize, LLVMDIFlagZero, LLVMDWARFEmissionKind, LLVMDWARFSourceLanguage,
        LLVMDebugMetadataVersion, LLVMDisposeDIBuilder, LLVMInstructionSetDebugLoc,
        LLVMSetSubprogram,
    },
    prelude::{LLVMDIBuilderRef, LLVMMetadataRef, LLVMValueRef},
};
use pliron::{
    graph::walkers::uninterruptible::immutable::walk_op,
    location::{Location, Source},
    uniqued_any,
};
use pliron_llvm::metadata::{
    MdNodeAttr, MdOperandAttr, attach_metadata, get_metadata_table, set_metadata_table,
};
use std::collections::HashMap;

/// The metadata kind that carries a location index from pliron to LLVM.
const LOCATION_KIND: &str = "cubecl.loc";

/// One frame of a location: a function, and the position in it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Frame {
    /// The `#[cube]` function, or `None` for a position outside a named function.
    pub name: Option<String>,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// The distinct locations of a module. A location is its frames, the innermost first.
#[derive(Debug, Default)]
pub(crate) struct LocationTable {
    locations: Vec<Vec<Frame>>,
    index: HashMap<Vec<Frame>, usize>,
}

impl LocationTable {
    fn intern(&mut self, frames: Vec<Frame>) -> usize {
        if let Some(index) = self.index.get(&frames) {
            return *index;
        }
        let index = self.locations.len();
        self.index.insert(frames.clone(), index);
        self.locations.push(frames);
        index
    }

    fn get(&self, index: usize) -> Option<&[Frame]> {
        self.locations.get(index).map(Vec::as_slice)
    }
}

/// The frames of `loc`, the innermost first. Empty for an unknown location.
fn frames(ctx: &Context, loc: &Location) -> Vec<Frame> {
    let mut frames = Vec::new();
    let mut loc = loc;
    loop {
        match loc {
            Location::CallSite { callee, caller } => {
                frames.extend(frame(ctx, callee));
                loc = caller;
            }
            other => {
                frames.extend(frame(ctx, other));
                return frames;
            }
        }
    }
}

/// The frame of a location that is not a call site.
fn frame(ctx: &Context, loc: &Location) -> Option<Frame> {
    match loc {
        Location::Named { name, child_loc } => Some(Frame {
            name: Some(name.clone()),
            ..frame(ctx, child_loc)?
        }),
        Location::SrcPos { src, pos } => Some(Frame {
            name: None,
            file: match src {
                Source::File(key) => uniqued_any::get(ctx, *key).display().to_string(),
                Source::InMemory => "<memory>".to_string(),
            },
            line: pos.line.max(0) as u32,
            column: pos.column.max(0) as u32,
        }),
        Location::Fused { locations, .. } => locations.iter().find_map(|loc| frame(ctx, loc)),
        Location::CallSite { callee, .. } => frame(ctx, callee),
        Location::Unknown => None,
    }
}

/// Ops that convert to a global, not to an instruction.
const NOT_INSTRUCTIONS: &[&str] = &["llvm.func", "llvm.global"];

/// Ops that LLVM's builder always gives as a constant.
const CONSTANT_OPS: &[&str] = &[
    "llvm.constant",
    "llvm.zero",
    "llvm.undef",
    "llvm.poison",
    "llvm.addressof",
    "llvm.blockaddress",
];

/// Ops that LLVM's builder folds to a constant when all operands are constants.
const FOLDABLE_OPS: &[&str] = &[
    "llvm.add",
    "llvm.sub",
    "llvm.mul",
    "llvm.udiv",
    "llvm.sdiv",
    "llvm.urem",
    "llvm.srem",
    "llvm.and",
    "llvm.or",
    "llvm.xor",
    "llvm.shl",
    "llvm.lshr",
    "llvm.ashr",
    "llvm.fadd",
    "llvm.fsub",
    "llvm.fmul",
    "llvm.fdiv",
    "llvm.frem",
    "llvm.fneg",
    "llvm.icmp",
    "llvm.fcmp",
    "llvm.select",
    "llvm.gep",
    "llvm.extractelement",
    "llvm.insertelement",
    "llvm.extract_value",
    "llvm.insert_value",
    "llvm.shuffle_vector",
    "llvm.addrspacecast",
    "llvm.bitcast",
    "llvm.fpext",
    "llvm.fptosi",
    "llvm.fptoui",
    "llvm.fptrunc",
    "llvm.inttoptr",
    "llvm.ptrtoint",
    "llvm.sext",
    "llvm.sitofp",
    "llvm.trunc",
    "llvm.uitofp",
    "llvm.zext",
];

/// Attaches `cubecl.loc` to each op of `module` that has a location and converts to an
/// instruction. An op that LLVM's builder folds to a constant gets none: `pliron-llvm` would log a
/// warning for it, and LLVM drops the metadata of a folded op anyway.
pub(crate) fn encode_locations(ctx: &Context, module: ModuleOp) -> LocationTable {
    let mut ops = Vec::new();
    walk_op(
        ctx,
        &mut ops,
        &WALKCONFIG_PREORDER_FORWARD,
        module.get_operation(),
        |_ctx, ops, node| {
            if let IRNode::Operation(op) = node {
                ops.push(op);
            }
        },
    );

    let mut table = LocationTable::default();
    let mut metadata = get_metadata_table(ctx, module).unwrap_or_default();
    let mut constants = Vec::new();
    let mut attachments = Vec::new();
    for op in ops {
        let name = Operation::get_opid(op, ctx).to_string();
        let operation = op.deref(ctx);
        let folds = CONSTANT_OPS.contains(&name.as_str())
            || (FOLDABLE_OPS.contains(&name.as_str())
                && operation.operands().all(|value| {
                    value
                        .defining_op()
                        .is_some_and(|def| constants.contains(&def))
                }));
        if folds {
            constants.push(op);
            continue;
        }
        let frames = frames(ctx, &operation.loc());
        if frames.is_empty()
            || !name.starts_with("llvm.")
            || NOT_INSTRUCTIONS.contains(&name.as_str())
        {
            continue;
        }
        let index = table.intern(frames);
        let node = metadata.push_uniqued(MdNodeAttr::new_tuple(vec![MdOperandAttr::String(
            index.to_string(),
        )]));
        attachments.push((op, node));
    }
    if attachments.is_empty() {
        return table;
    }
    set_metadata_table(ctx, module, metadata);
    for (op, node) in attachments {
        attach_metadata(ctx, op, LOCATION_KIND, node);
    }
    table
}

/// Replaces each `cubecl.loc` in `module` with a `!dbg` location from `table`, and adds the
/// subprograms, the compile unit and the module flags that DWARF needs.
pub(crate) fn attach_debug_info(module: &LlvmModule, table: &LocationTable, level: DebugInfo) {
    if level == DebugInfo::None || table.locations.is_empty() {
        return;
    }
    let ctx = module.context();
    // SAFETY: the builder and every node it creates belong to the module and its context, and the
    // builder is finalized and disposed before the module is used again.
    unsafe {
        let kind = LLVMGetMDKindIDInContext(
            ctx,
            LOCATION_KIND.as_ptr() as *const _,
            LOCATION_KIND.len() as u32,
        );
        let mut dwarf = Dwarf::new(module, table, level);
        let mut func = LLVMGetFirstFunction(module.raw());
        while !func.is_null() {
            dwarf.attach_function(func, kind);
            func = LLVMGetNextFunction(func);
        }
        dwarf.finish();
    }
}

/// The DWARF of one module while it is built.
struct Dwarf<'a> {
    module: &'a LlvmModule,
    table: &'a LocationTable,
    builder: LLVMDIBuilderRef,
    unit: LLVMMetadataRef,
    subroutine_type: LLVMMetadataRef,
    files: HashMap<String, LLVMMetadataRef>,
    /// The subprogram of each inlined function, by name and file.
    inlined: HashMap<(String, String), LLVMMetadataRef>,
}

impl<'a> Dwarf<'a> {
    unsafe fn new(module: &'a LlvmModule, table: &'a LocationTable, level: DebugInfo) -> Self {
        unsafe {
            let builder = LLVMCreateDIBuilder(module.raw());
            let first = &table.locations[0][0];
            let mut files = HashMap::new();
            let unit_file = file(builder, &mut files, &first.file);
            let producer = "cubecl";
            let kind = match level {
                DebugInfo::Full => LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
                _ => LLVMDWARFEmissionKind::LLVMDWARFEmissionKindLineTablesOnly,
            };
            let unit = LLVMDIBuilderCreateCompileUnit(
                builder,
                LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageRust,
                unit_file,
                producer.as_ptr() as *const _,
                producer.len(),
                1,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                0,
                kind,
                0,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
            );
            let subroutine_type = LLVMDIBuilderCreateSubroutineType(
                builder,
                unit_file,
                std::ptr::null_mut(),
                0,
                LLVMDIFlagZero,
            );
            Self {
                module,
                table,
                builder,
                unit,
                subroutine_type,
                files,
                inlined: HashMap::new(),
            }
        }
    }

    /// Gives each instruction of `func` its location. Each instruction without `cubecl.loc` gets
    /// the location of the function, because LLVM rejects a call without a location in a function
    /// with debug data.
    unsafe fn attach_function(&mut self, func: LLVMValueRef, kind: u32) {
        unsafe {
            let instructions = instructions(func);
            let located = instructions
                .iter()
                .filter_map(|inst| Some((*inst, self.location_index(*inst, kind)?)))
                .collect::<Vec<_>>();
            // The outermost frame that comes first in the source is the function definition.
            let Some(root) = located
                .iter()
                .filter_map(|(_, index)| self.table.get(*index)?.last())
                .min_by_key(|frame| (frame.line, frame.column))
            else {
                return;
            };

            let mut len = 0;
            let linkage = LLVMGetValueName2(func, &mut len);
            let name = root.name.as_deref().unwrap_or("kernel");
            let file = file(self.builder, &mut self.files, &root.file);
            let subprogram = LLVMDIBuilderCreateFunction(
                self.builder,
                file,
                name.as_ptr() as *const _,
                name.len(),
                linkage,
                len,
                file,
                root.line,
                self.subroutine_type,
                0,
                1,
                root.line,
                LLVMDIFlagZero,
                1,
            );
            LLVMSetSubprogram(func, subprogram);
            let entry = LLVMDIBuilderCreateDebugLocation(
                self.module.context(),
                root.line,
                0,
                subprogram,
                std::ptr::null_mut(),
            );

            let mut locations = HashMap::new();
            for inst in instructions {
                let loc = match self.location_index(inst, kind) {
                    Some(index) => *locations
                        .entry(index)
                        .or_insert_with(|| self.location(index, subprogram).unwrap_or(entry)),
                    None => entry,
                };
                LLVMInstructionSetDebugLoc(inst, loc);
                LLVMSetMetadata(inst, kind, std::ptr::null_mut());
            }
        }
    }

    /// The index in `cubecl.loc` of `inst`.
    unsafe fn location_index(&self, inst: LLVMValueRef, kind: u32) -> Option<usize> {
        unsafe {
            let node = LLVMGetMetadata(inst, kind);
            if node.is_null() || LLVMGetMDNodeNumOperands(node) != 1 {
                return None;
            }
            let mut operand = std::ptr::null_mut();
            LLVMGetMDNodeOperands(node, &mut operand);
            let mut len = 0;
            let text = LLVMGetMDString(operand, &mut len);
            if text.is_null() {
                return None;
            }
            let text = std::slice::from_raw_parts(text as *const u8, len as usize);
            std::str::from_utf8(text).ok()?.parse().ok()
        }
    }

    /// The `DILocation` of location `index` in the function with subprogram `subprogram`. The
    /// outermost frame is the function itself. Each inner frame is inlined at the one outside it.
    unsafe fn location(
        &mut self,
        index: usize,
        subprogram: LLVMMetadataRef,
    ) -> Option<LLVMMetadataRef> {
        let frames = self.table.get(index)?;
        let mut inlined_at = std::ptr::null_mut();
        for (depth, frame) in frames.iter().rev().enumerate() {
            let scope = if depth == 0 {
                subprogram
            } else {
                // SAFETY: the caller's contract.
                unsafe { self.inlined_subprogram(frame) }
            };
            // SAFETY: the scope and `inlined_at` belong to the module's context.
            inlined_at = unsafe {
                LLVMDIBuilderCreateDebugLocation(
                    self.module.context(),
                    frame.line,
                    frame.column,
                    scope,
                    inlined_at,
                )
            };
        }
        Some(inlined_at)
    }

    /// The subprogram of the inlined function of `frame`.
    unsafe fn inlined_subprogram(&mut self, frame: &Frame) -> LLVMMetadataRef {
        let name = frame
            .name
            .clone()
            .unwrap_or_else(|| "<inlined>".to_string());
        let key = (name, frame.file.clone());
        if let Some(subprogram) = self.inlined.get(&key) {
            return *subprogram;
        }
        // SAFETY: the builder is live.
        let subprogram = unsafe {
            let file = file(self.builder, &mut self.files, &frame.file);
            LLVMDIBuilderCreateFunction(
                self.builder,
                file,
                key.0.as_ptr() as *const _,
                key.0.len(),
                std::ptr::null(),
                0,
                file,
                frame.line,
                self.subroutine_type,
                1,
                1,
                frame.line,
                LLVMDIFlagZero,
                1,
            )
        };
        self.inlined.insert(key, subprogram);
        subprogram
    }

    /// Adds the module flags and finalizes the builder.
    unsafe fn finish(self) {
        let _ = self.unit;
        unsafe {
            let ctx = self.module.context();
            for (key, value) in [
                ("Debug Info Version", LLVMDebugMetadataVersion()),
                ("Dwarf Version", 4),
            ] {
                let value = LLVMConstInt(LLVMInt32TypeInContext(ctx), value as u64, 0);
                LLVMAddModuleFlag(
                    self.module.raw(),
                    LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning,
                    key.as_ptr() as *const _,
                    key.len(),
                    LLVMValueAsMetadata(value),
                );
            }
            LLVMDIBuilderFinalize(self.builder);
            LLVMDisposeDIBuilder(self.builder);
        }
    }
}

/// The `DIFile` of `path`. The path is as `file!()` gives it.
unsafe fn file(
    builder: LLVMDIBuilderRef,
    files: &mut HashMap<String, LLVMMetadataRef>,
    path: &str,
) -> LLVMMetadataRef {
    *files.entry(path.to_string()).or_insert_with(|| {
        // SAFETY: the caller's contract, and `path` is read for the length given.
        unsafe {
            LLVMDIBuilderCreateFile(
                builder,
                path.as_ptr() as *const _,
                path.len(),
                c"".as_ptr(),
                0,
            )
        }
    })
}

/// The instructions of `func`, in order.
unsafe fn instructions(func: LLVMValueRef) -> Vec<LLVMValueRef> {
    let mut out = Vec::new();
    unsafe {
        let mut block = LLVMGetFirstBasicBlock(func);
        while !block.is_null() {
            let mut inst = LLVMGetFirstInstruction(block);
            while !inst.is_null() {
                out.push(inst);
                inst = LLVMGetNextInstruction(inst);
            }
            block = LLVMGetNextBasicBlock(block);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cpu::jit::engine::to_llvm_module,
        shared::{PlironOptions, base::lower_cpu, offline_kernels::device_properties},
    };
    use cubecl_core as cubecl;
    use cubecl_core::prelude::*;
    use cubecl_runtime::kernel::CubeKernel;
    use std::sync::Arc;

    #[cube]
    fn inner(x: f32) -> f32 {
        let y = x * x;
        y / 3.0
    }

    #[cube]
    fn mid(x: f32) -> f32 {
        inner(x) * 2.0
    }

    #[cube(launch)]
    fn outer(input: &[f32], output: &mut [f32]) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = mid(input[ABSOLUTE_POS]);
        }
    }

    fn lowered(level: DebugInfo) -> super::super::base::LoweredCpu {
        let settings = KernelSettings::new(
            *CubeDim::new_1d(64),
            ExecutionMode::Checked,
            AddressType::U32,
        )
        .debug_info(level);
        let kernel = outer::Outer::new(
            settings,
            device_properties(32),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        lower_cpu(kernel.define(), &PlironOptions::default()).unwrap()
    }

    /// Each float op and each store without a location. These ops come from the kernel source.
    /// The entry ABI adds the loops over the units, which have no source and get the kernel
    /// entry location in the DWARF.
    fn unlocated_source_ops(lowered: &super::super::base::LoweredCpu) -> Vec<String> {
        let mut ops = Vec::new();
        walk_op(
            &lowered.ctx,
            &mut ops,
            &WALKCONFIG_PREORDER_FORWARD,
            lowered.module.get_operation(),
            |ctx, ops, node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let name = Operation::get_opid(op, ctx).to_string();
                let from_source = name == "llvm.store"
                    || name.starts_with("llvm.call")
                    || (name.starts_with("llvm.f")
                        && !["llvm.func", "llvm.fence"].contains(&name.as_str()));
                if from_source && op.deref(ctx).loc().is_unknown() {
                    ops.push(name);
                }
            },
        );
        ops
    }

    /// The rewrites and the fallback pass keep a location on each op that comes from source.
    #[test]
    fn lowering_keeps_source_locations() {
        let lowered = lowered(DebugInfo::LineTables);
        let unlocated = unlocated_source_ops(&lowered);
        assert!(
            unlocated.is_empty(),
            "ops without a location: {unlocated:?}"
        );

        // `x * x` and `y / 3.0` in `inner` keep their own positions. The fallback pass alone would
        // give both the position of the op before them.
        let mut ops = Vec::new();
        walk_op(
            &lowered.ctx,
            &mut ops,
            &WALKCONFIG_PREORDER_FORWARD,
            lowered.module.get_operation(),
            |_ctx, ops, node| {
                if let IRNode::Operation(op) = node {
                    ops.push(op);
                }
            },
        );
        let mut positions = ops
            .into_iter()
            .map(|op| frames(&lowered.ctx, &op.deref(&lowered.ctx).loc()))
            .filter_map(|frames| frames.into_iter().next())
            .filter(|frame| frame.name.as_deref() == Some("inner"))
            .map(|frame| (frame.line, frame.column))
            .collect::<Vec<_>>();
        positions.sort();
        positions.dedup();
        assert!(positions.len() >= 2, "positions in `inner`: {positions:?}");
    }

    #[test]
    fn inlined_functions_are_dwarf_frames() {
        let lowered = lowered(DebugInfo::LineTables);
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            lowered.debug_info,
        )
        .unwrap();
        module.verify().unwrap();
        let ir = module.print();
        for name in ["outer", "mid", "inner"] {
            assert!(
                ir.contains(&format!("DISubprogram(name: \"{name}\"")),
                "{ir}"
            );
        }
        assert!(ir.contains("inlinedAt:"), "{ir}");
        assert!(ir.contains("emissionKind: LineTablesOnly"), "{ir}");
        assert!(
            !ir.contains(LOCATION_KIND),
            "the bridge metadata is removed:\n{ir}"
        );
    }

    /// The ops have locations in a `dev` build. The level `None` must still give no debug data.
    #[test]
    fn no_debug_data_without_debug_info() {
        let lowered = lowered(DebugInfo::None);
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            DebugInfo::None,
        )
        .unwrap();
        let ir = module.print();
        assert!(
            !ir.contains("!dbg") && !ir.contains("DICompileUnit"),
            "{ir}"
        );
    }
}
