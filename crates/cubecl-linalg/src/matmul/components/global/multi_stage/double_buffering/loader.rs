use std::marker::PhantomData;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::buffer_loading::BufferLoading;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CommonGlobalConfig, InputLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::stage::{TilingLayout, TilingOrder};
use crate::matmul::components::{global, Ident, InvalidConfigError};
use crate::tensor::VirtualTensor;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;

#[derive(CubeType)]
pub struct LhsBufferLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig> InputLoader<EG, ES, CommonGlobalConfig<S>>
    for LhsBufferLoader<EG, ES, S>
{
    type StageReader = LhsBufferReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        load_buffer::<EG, ES, S>(
            this.buffer_iter,
            &this.tensor_view,
            &mut this.stage,
            Ident::Lhs,
            config,
        );
    }

    fn fill_stage_window(
        _this: &mut Self,
        _pipeline: Pipeline<ES>,
        #[comptime] _config: CommonGlobalConfig<S>,
    ) {
        comptime!(todo!());
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig> LhsBufferLoader<EG, ES, S> {
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsBufferLoader::<EG, ES, S> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.stage_tiling(Ident::Lhs).tile_count_col(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig> InputLoader<EG, ES, CommonGlobalConfig<S>>
    for RhsBufferLoader<EG, ES, S>
{
    type StageReader = RhsBufferReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        load_buffer::<EG, ES, S>(
            this.buffer_iter,
            &this.tensor_view,
            &mut this.stage,
            Ident::Rhs,
            config,
        );
    }

    fn fill_stage_window(
        _this: &mut Self,
        _pipeline: Pipeline<ES>,
        #[comptime] _config: CommonGlobalConfig<S>,
    ) {
        comptime!(todo!());
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        RhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig> RhsBufferLoader<EG, ES, S> {
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsBufferLoader::<EG, ES, S> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.stage_tiling(Ident::Rhs).tile_count_row(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
fn load_buffer<EG: Numeric, ES: Numeric, S: stage::StageConfig>(
    buffer_iter: u32,
    tensor_view: &TensorReader<EG>,
    stage: &mut Stage<ES>,
    #[comptime] ident: Ident,
    #[comptime] config: CommonGlobalConfig<S>,
) {
    let buffer_num_elements = config.stage_tiling(ident).buffer_size(ident.as_input());
    let line_size = config.stage_line_size(ident);
    let buffer_num_lines = buffer_num_elements / line_size;

    let start = buffer_iter * buffer_num_lines;
    let end = start + buffer_num_lines;
    let buffer_slice = &mut stage.as_slice_mut().slice_mut(start, end);

    BufferLoading::load_to_slice::<EG, ES, CommonGlobalConfig<S>>(
        tensor_view,
        buffer_slice,
        config.num_planes(),
        0u32,
        ident,
        config,
    );
}

pub fn check_buffers_contiguous<G: global::GlobalConfig>(
    ident: Ident,
    config: &G,
) -> Result<(), InvalidConfigError> {
    match ident.as_input() {
        InputIdent::Lhs => {
            if let TilingLayout::Contiguous(TilingOrder::RowMajor) = config.tiling_layout(ident) {
                return Err(Box::new(
                    "Lhs must have ColMajor tiling order in pipelined setting",
                ));
            }
        }
        InputIdent::Rhs => {
            if let TilingLayout::Contiguous(TilingOrder::ColMajor) = config.tiling_layout(ident) {
                return Err(Box::new(
                    "Rhs must have RowMajor tiling order in pipelined setting",
                ));
            }
        }
    }

    Ok(())
}
