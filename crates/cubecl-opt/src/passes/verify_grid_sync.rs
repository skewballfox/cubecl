//! Refuses a grid sync that some units of the launch could skip. Such a grid sync hangs the
//! device, because the units that reach it wait for units that never arrive.
//!
//! The check is conservative: it refuses what the uniformity analysis cannot prove. The analysis
//! tracks loop-carried values only to cube scope, so it proves `for` loops with uniform bounds,
//! but not `while` loops. `sync_grid_unchecked` skips the check.
//!
//! Run it after `Mem2RegPass`: before it, a condition is a load from a local variable, and the
//! uniformity analysis reads every load as non-uniform.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use cubecl_ir::{
    attributes::EntrypointInterface,
    dialect::{BlockPtrExt, branch::ConditionOp, synchronization::GridSyncOp},
    interfaces::uniformity::Uniformity,
    prelude::*,
};
use pliron::{builtin::ops::FuncOp, linked_list::ContainsLinkedList, verify_err};
use thiserror::Error;

use crate::{
    analyses::dataflow_solver::value_uniformity::DynamicUniformityLattice,
    passes::uniformity::UniformityAnalysis,
};

#[derive(Error, Debug)]
pub enum GridSyncError {
    #[error(
        "`sync_grid` must be reached by every unit, but the control flow of `{_0}` can differ \
         between units. Move `sync_grid` out of it, or use a `for` loop with the same bounds for \
         all units."
    )]
    NotUniform(String),
    #[error("`sync_grid` is inside a function that is not inlined into the kernel: `{_0}`")]
    NotInlined(String),
}

/// Verifies that every unit of the launch reaches each grid sync.
pub struct VerifyGridSyncPass;

#[pass_name]
impl Pass for VerifyGridSyncPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut syncs = Vec::new();
        visit_all_ops_of_type::<GridSyncOp, _>(ctx, &mut syncs, op, |ctx, syncs, sync| {
            if sync.checked(ctx).0 {
                syncs.push(sync.get_operation())
            }
        });

        let mut res = PassResult::default();
        res.set_preserved::<UniformityAnalysis>();
        if syncs.is_empty() {
            return Ok(res);
        }

        let analysis = analyses.get_analysis::<UniformityAnalysis>(op, ctx)?;
        for sync in syncs {
            verify_reached_by_all(ctx, &analysis, sync)?;
        }
        Ok(res)
    }
}

/// Walks from `sync` up to the entry function. Every op on the way that selects which region
/// runs, and every loop condition inside it, must be the same for all units of the launch.
fn verify_reached_by_all(
    ctx: &Context,
    analysis: &UniformityAnalysis,
    sync: Ptr<Operation>,
) -> Result<()> {
    let is_device_uniform = |value: Value| {
        analysis
            .lookup_state::<DynamicUniformityLattice>(value)
            .is_some_and(|lattice| lattice.deref().value().0 == Uniformity::Device)
    };
    let loc = sync.deref(ctx).loc();
    let mut current = sync;

    while let Some(block) = current.deref(ctx).get_parent_block() {
        let Some(parent) = current.deref(ctx).get_parent_op(ctx) else {
            break;
        };
        if let Some(func) = Operation::get_op::<FuncOp>(parent, ctx) {
            if func.get_entrypoint_abi(ctx).is_some() && block.is_entry_block(ctx) {
                return Ok(());
            }
            return verify_err!(
                loc,
                GridSyncError::NotInlined(func.get_symbol_name(ctx).to_string())
            );
        }

        let mut deciders: Vec<Value> = parent.deref(ctx).operands().collect();
        for region in parent.deref(ctx).regions() {
            let terminators = region
                .deref(ctx)
                .iter(ctx)
                .filter_map(|block| block.deref(ctx).get_terminator(ctx));
            deciders.extend(
                terminators
                    .filter_map(|term| Operation::get_op::<ConditionOp>(term, ctx))
                    .map(|condition| condition.condition(ctx)),
            );
        }
        if !block.is_entry_block(ctx) || !deciders.into_iter().all(is_device_uniform) {
            return verify_err!(
                loc,
                GridSyncError::NotUniform(parent.dyn_op(ctx).get_opid().to_string())
            );
        }
        current = parent;
    }
    Ok(())
}
