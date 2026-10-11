//! A grid sync that some units of the launch skip hangs the device. The verifier must refuse
//! every kernel where control flow that differs between units decides if a grid sync runs, and
//! accept the kernels where all units take the same path.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    dialect::scf::BranchToSCFPass,
    settings::{CooperativeOptions, Dim3, ExecutionMode, Persistence},
};
use cubecl_opt::passes::{
    mem2reg::Mem2RegPass, sroa::SROAPass, verify_grid_sync::VerifyGridSyncPass,
};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    op::Op,
    pass::{AnalysisManager, NestedOpsPass, OpPass, Pass, Passes},
};

#[cube]
fn at_top_level() {
    sync_grid();
}

#[cube]
fn unchecked_in_unit_branch() {
    if UNIT_POS == 0 {
        unsafe { sync_grid_unchecked() };
    }
}

/// Uniform, but the analysis tracks loop-carried values only to cube scope.
#[cube]
fn in_uniform_while_loop() {
    let mut i = 0;
    while i < CUBE_COUNT {
        sync_grid();
        i += 1;
    }
}

#[cube]
fn in_uniform_branch() {
    if CUBE_COUNT > 1 {
        sync_grid();
    }
}

#[cube]
fn in_uniform_loop() {
    for _ in 0..CUBE_COUNT {
        sync_grid();
    }
}

#[cube]
fn in_unit_branch() {
    if UNIT_POS == 0 {
        sync_grid();
    }
}

#[cube]
fn in_cube_branch() {
    if CUBE_POS == 0 {
        sync_grid();
    }
}

#[cube]
fn in_unit_loop() {
    let mut i = 0;
    while i < UNIT_POS {
        sync_grid();
        i += 1;
    }
}

#[cube]
fn after_unit_return() {
    if UNIT_POS == 0 {
        terminate!();
    }
    sync_grid();
}

/// Expands `body` in a cooperative kernel, promotes locals to registers, then verifies.
fn verify(body: fn(&Scope)) -> Result<(), String> {
    let settings = KernelSettings::new(Dim3::new_1d(32), ExecutionMode::Checked, AddressType::U32)
        .persistence(Persistence::Cooperative(CooperativeOptions::default()));
    let scope = Scope::root(settings);
    body(&scope);
    let errors = scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");

    let module_op = scope.state().module.get_operation();
    let mut ctx = scope.into_context().expect("the scope owns its context");
    let mut func_passes = OpPass::<FuncOp, Passes>::default();
    func_passes.add_pass(SROAPass);
    func_passes.add_pass(BranchToSCFPass::default());
    func_passes.add_pass(Mem2RegPass);
    let mut passes = OpPass::<ModuleOp, Passes>::default();
    passes.add_pass(NestedOpsPass::new(func_passes));
    passes.add_pass(VerifyGridSyncPass);

    passes
        .run(module_op, &mut ctx, &mut AnalysisManager::default())
        .map(|_| ())
        .map_err(|err| err.to_string())
}

#[test]
fn accepts_control_flow_that_is_the_same_for_all_units() {
    verify(at_top_level::expand).expect("top level");
    verify(in_uniform_branch::expand).expect("uniform branch");
    verify(in_uniform_loop::expand).expect("uniform loop");
}

#[test]
fn skips_unchecked_grid_syncs() {
    verify(unchecked_in_unit_branch::expand).expect("unchecked");
}

#[test]
fn refuses_control_flow_that_differs_between_units() {
    for (name, body) in [
        ("unit branch", in_unit_branch::expand as fn(&Scope)),
        ("cube branch", in_cube_branch::expand),
        ("unit loop", in_unit_loop::expand),
        ("unit return", after_unit_return::expand),
        ("while loop it cannot prove", in_uniform_while_loop::expand),
    ] {
        let error = verify(body).expect_err(name);
        assert!(
            error.contains("must be reached by every unit"),
            "{name}: {error}"
        );
    }
}
