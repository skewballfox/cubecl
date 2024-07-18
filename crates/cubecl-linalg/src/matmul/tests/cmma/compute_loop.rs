use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{
    base::{make_accumulators, SharedMemories, SharedMemoriesExpand},
    compute_loop::compute_loop,
    config::CmmaConfig,
};
use crate::matmul::tests::test_utils::{
    assert_equals, cmma_available, create_empty, range_tensor_f16,
};

#[cube(launch)]
fn compute_loop_test<F: Float, FC: Float>(
    lhs_tensor: &Tensor<FC>,
    rhs_tensor: &Tensor<FC>,
    accumulate_array: &mut Array<F>,
    m: Comptime<UInt>,
    k: Comptime<UInt>,
    n: Comptime<UInt>,
    config: Comptime<CmmaConfig>,
) {
    let mut lhs = SharedMemory::<FC>::new(Comptime::get(m * k));
    let mut rhs = SharedMemory::<FC>::new(Comptime::get(k * n));

    for i in range(0u32, Comptime::get(m * k), Comptime::new(false)) {
        lhs[i] = lhs_tensor[i];
    }
    for i in range(0u32, Comptime::get(k * n), Comptime::new(false)) {
        rhs[i] = rhs_tensor[i];
    }
    for i in range(0u32, Comptime::get(m * n), Comptime::new(false)) {
        accumulate_array[i] = F::new(0.);
    }

    let shared_memories = SharedMemories { lhs, rhs };
    let accumulators = make_accumulators::<F>();

    compute_loop(shared_memories, accumulators, config);

    let offset = UNIT_POS_Y * UInt::new(512);
    let slice_0 = accumulate_array.slice_mut(offset, offset + UInt::new(256));
    cmma::store::<F>(
        slice_0,
        &accumulators.first,
        UInt::new(16),
        cmma::MatrixLayout::RowMajor,
    );

    let slice_1 = accumulate_array.slice_mut(offset + UInt::new(256), offset + UInt::new(512));
    cmma::store::<F>(
        slice_1,
        &accumulators.second,
        UInt::new(16),
        cmma::MatrixLayout::RowMajor,
    );
}

/// Exported test
pub fn compute_loop_k_test<R: Runtime>(device: &R::Device) {
    if !cmma_available::<R>(device) {
        // We can't execute the test, skip.
        return;
    }

    let m = 16;
    let k = 32;
    let n = 16;
    let lhs = range_tensor_f16::<R>(m, k, device);
    let rhs = range_tensor_f16::<R>(k, n, device);
    let results = create_empty::<R>(m, n, device);
    let cube_dim = CubeDim::new(32, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = CmmaConfig {
        block_size_m: UInt::new(m as u32),
        block_size_k: UInt::new(k as u32),
        block_size_n: UInt::new(n as u32),
        tile_size: UInt::new(16),
        check_m_bounds: false,
        check_k_bounds: false,
        check_n_bounds: false,
        unroll: false,
    };

    compute_loop_test::launch::<F32, F16, R>(
        R::client(device),
        cube_count,
        cube_dim,
        TensorArg::new(&lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::new(&rhs.handle, &rhs.strides, &rhs.shape),
        ArrayArg::new(&results, m * n),
        UInt::new(m as u32),
        UInt::new(k as u32),
        UInt::new(n as u32),
        config,
    );

    let expected = &[
        1610496., 1614832., 1619168., 1623504., 1627840., 1632176., 1636512., 1640848., 1645184.,
        1649520., 1653856., 1658192., 1662528., 1666864., 1671200., 1675536., 1737472., 1742320.,
        1747168., 1752016., 1756864., 1761712., 1766560., 1771408., 1776256., 1781104., 1785952.,
        1790800., 1795648., 1800496., 1805344., 1810192., 1864448., 1869808., 1875168., 1880528.,
        1885888., 1891248., 1896608., 1901968., 1907328., 1912688., 1918048., 1923408., 1928768.,
        1934128., 1939488., 1944848., 1991424., 1997296., 2003168., 2009040., 2014912., 2020784.,
        2026656., 2032528., 2038400., 2044272., 2050144., 2056016., 2061888., 2067760., 2073632.,
        2079504., 2118400., 2124784., 2131168., 2137552., 2143936., 2150320., 2156704., 2163088.,
        2169472., 2175856., 2182240., 2188624., 2195008., 2201392., 2207776., 2214160., 2245376.,
        2252272., 2259168., 2266064., 2272960., 2279856., 2286752., 2293648., 2300544., 2307440.,
        2314336., 2321232., 2328128., 2335024., 2341920., 2348816., 2372352., 2379760., 2387168.,
        2394576., 2401984., 2409392., 2416800., 2424208., 2431616., 2439024., 2446432., 2453840.,
        2461248., 2468656., 2476064., 2483472., 2499328., 2507248., 2515168., 2523088., 2531008.,
        2538928., 2546848., 2554768., 2562688., 2570608., 2578528., 2586448., 2594368., 2602288.,
        2610208., 2618128., 2626304., 2634736., 2643168., 2651600., 2660032., 2668464., 2676896.,
        2685328., 2693760., 2702192., 2710624., 2719056., 2727488., 2735920., 2744352., 2752784.,
        2753280., 2762224., 2771168., 2780112., 2789056., 2798000., 2806944., 2815888., 2824832.,
        2833776., 2842720., 2851664., 2860608., 2869552., 2878496., 2887440., 2880256., 2889712.,
        2899168., 2908624., 2918080., 2927536., 2936992., 2946448., 2955904., 2965360., 2974816.,
        2984272., 2993728., 3003184., 3012640., 3022096., 3007232., 3017200., 3027168., 3037136.,
        3047104., 3057072., 3067040., 3077008., 3086976., 3096944., 3106912., 3116880., 3126848.,
        3136816., 3146784., 3156752., 3134208., 3144688., 3155168., 3165648., 3176128., 3186608.,
        3197088., 3207568., 3218048., 3228528., 3239008., 3249488., 3259968., 3270448., 3280928.,
        3291408., 3261184., 3272176., 3283168., 3294160., 3305152., 3316144., 3327136., 3338128.,
        3349120., 3360112., 3371104., 3382096., 3393088., 3404080., 3415072., 3426064., 3388160.,
        3399664., 3411168., 3422672., 3434176., 3445680., 3457184., 3468688., 3480192., 3491696.,
        3503200., 3514704., 3526208., 3537712., 3549216., 3560720., 3515136., 3527152., 3539168.,
        3551184., 3563200., 3575216., 3587232., 3599248., 3611264., 3623280., 3635296., 3647312.,
        3659328., 3671344., 3683360., 3695376.,
    ];

    assert_equals::<R>(results, expected, device);
}

/// Exported test
pub fn compute_loop_warp_test<R: Runtime>(device: &R::Device) {
    if !cmma_available::<R>(device) {
        // We can't execute the test, skip.
        return;
    }

    let m = 16;
    let k = 32;
    let n = 32;
    let lhs = range_tensor_f16::<R>(m, k, device);
    let rhs = range_tensor_f16::<R>(k, n, device);
    let results = create_empty::<R>(m, n, device);
    let cube_dim = CubeDim::new(32, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = CmmaConfig {
        block_size_m: UInt::new(m as u32),
        block_size_k: UInt::new(k as u32),
        block_size_n: UInt::new(n as u32),
        tile_size: UInt::new(16),
        check_m_bounds: false,
        check_k_bounds: false,
        check_n_bounds: false,
        unroll: false,
    };

    compute_loop_test::launch::<F32, F16, R>(
        R::client(device),
        cube_count,
        cube_dim,
        TensorArg::new(&lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::new(&rhs.handle, &rhs.strides, &rhs.shape),
        ArrayArg::new(&results, m * n),
        UInt::new(m as u32),
        UInt::new(k as u32),
        UInt::new(n as u32),
        config,
    );

    let expected = &[
        1610496., 1614832., 1619168., 1623504., 1627840., 1632176., 1636512., 1640848., 1645184.,
        1649520., 1653856., 1658192., 1662528., 1666864., 1671200., 1675536., 1737472., 1742320.,
        1747168., 1752016., 1756864., 1761712., 1766560., 1771408., 1776256., 1781104., 1785952.,
        1790800., 1795648., 1800496., 1805344., 1810192., 1864448., 1869808., 1875168., 1880528.,
        1885888., 1891248., 1896608., 1901968., 1907328., 1912688., 1918048., 1923408., 1928768.,
        1934128., 1939488., 1944848., 1991424., 1997296., 2003168., 2009040., 2014912., 2020784.,
        2026656., 2032528., 2038400., 2044272., 2050144., 2056016., 2061888., 2067760., 2073632.,
        2079504., 2118400., 2124784., 2131168., 2137552., 2143936., 2150320., 2156704., 2163088.,
        2169472., 2175856., 2182240., 2188624., 2195008., 2201392., 2207776., 2214160., 2245376.,
        2252272., 2259168., 2266064., 2272960., 2279856., 2286752., 2293648., 2300544., 2307440.,
        2314336., 2321232., 2328128., 2335024., 2341920., 2348816., 2372352., 2379760., 2387168.,
        2394576., 2401984., 2409392., 2416800., 2424208., 2431616., 2439024., 2446432., 2453840.,
        2461248., 2468656., 2476064., 2483472., 2499328., 2507248., 2515168., 2523088., 2531008.,
        2538928., 2546848., 2554768., 2562688., 2570608., 2578528., 2586448., 2594368., 2602288.,
        2610208., 2618128., 2626304., 2634736., 2643168., 2651600., 2660032., 2668464., 2676896.,
        2685328., 2693760., 2702192., 2710624., 2719056., 2727488., 2735920., 2744352., 2752784.,
        2753280., 2762224., 2771168., 2780112., 2789056., 2798000., 2806944., 2815888., 2824832.,
        2833776., 2842720., 2851664., 2860608., 2869552., 2878496., 2887440., 2880256., 2889712.,
        2899168., 2908624., 2918080., 2927536., 2936992., 2946448., 2955904., 2965360., 2974816.,
        2984272., 2993728., 3003184., 3012640., 3022096., 3007232., 3017200., 3027168., 3037136.,
        3047104., 3057072., 3067040., 3077008., 3086976., 3096944., 3106912., 3116880., 3126848.,
        3136816., 3146784., 3156752., 3134208., 3144688., 3155168., 3165648., 3176128., 3186608.,
        3197088., 3207568., 3218048., 3228528., 3239008., 3249488., 3259968., 3270448., 3280928.,
        3291408., 3261184., 3272176., 3283168., 3294160., 3305152., 3316144., 3327136., 3338128.,
        3349120., 3360112., 3371104., 3382096., 3393088., 3404080., 3415072., 3426064., 3388160.,
        3399664., 3411168., 3422672., 3434176., 3445680., 3457184., 3468688., 3480192., 3491696.,
        3503200., 3514704., 3526208., 3537712., 3549216., 3560720., 3515136., 3527152., 3539168.,
        3551184., 3563200., 3575216., 3587232., 3599248., 3611264., 3623280., 3635296., 3647312.,
        3659328., 3671344., 3683360., 3695376., 3830528., 3834864., 3839200., 3843536., 3847872.,
        3852208., 3856544., 3860880., 3865216., 3869552., 3873888., 3878224., 3882560., 3886896.,
        3891232., 3895568., 4219648., 4224496., 4229344., 4234192., 4239040., 4243888., 4248736.,
        4253584., 4258432., 4263280., 4268128., 4272976., 4277824., 4282672., 4287520., 4292368.,
        4608768., 4614128., 4619488., 4624848., 4630208., 4635568., 4640928., 4646288., 4651648.,
        4657008., 4662368., 4667728., 4673088., 4678448., 4683808., 4689168., 4997888., 5003760.,
        5009632., 5015504., 5021376., 5027248., 5033120., 5038992., 5044864., 5050736., 5056608.,
        5062480., 5068352., 5074224., 5080096., 5085968., 5387008., 5393392., 5399776., 5406160.,
        5412544., 5418928., 5425312., 5431696., 5438080., 5444464., 5450848., 5457232., 5463616.,
        5470000., 5476384., 5482768., 5776128., 5783024., 5789920., 5796816., 5803712., 5810608.,
        5817504., 5824400., 5831296., 5838192., 5845088., 5851984., 5858880., 5865776., 5872672.,
        5879568., 6165248., 6172656., 6180064., 6187472., 6194880., 6202288., 6209696., 6217104.,
        6224512., 6231920., 6239328., 6246736., 6254144., 6261552., 6268960., 6276368., 6554368.,
        6562288., 6570208., 6578128., 6586048., 6593968., 6601888., 6609808., 6617728., 6625648.,
        6633568., 6641488., 6649408., 6657328., 6665248., 6673168., 6943488., 6951920., 6960352.,
        6968784., 6977216., 6985648., 6994080., 7002512., 7010944., 7019376., 7027808., 7036240.,
        7044672., 7053104., 7061536., 7069968., 7332608., 7341552., 7350496., 7359440., 7368384.,
        7377328., 7386272., 7395216., 7404160., 7413104., 7422048., 7430992., 7439936., 7448880.,
        7457824., 7466768., 7721728., 7731184., 7740640., 7750096., 7759552., 7769008., 7778464.,
        7787920., 7797376., 7806832., 7816288., 7825744., 7835200., 7844656., 7854112., 7863568.,
        8110848., 8120816., 8130784., 8140752., 8150720., 8160688., 8170656., 8180624., 8190592.,
        8200560., 8210528., 8220496., 8230464., 8240432., 8250400., 8260368., 8499968., 8510448.,
        8520928., 8531408., 8541888., 8552368., 8562848., 8573328., 8583808., 8594288., 8604768.,
        8615248., 8625728., 8636208., 8646688., 8657168., 8889088., 8900080., 8911072., 8922064.,
        8933056., 8944048., 8955040., 8966032., 8977024., 8988016., 8999008., 9010000., 9020992.,
        9031984., 9042976., 9053968., 9278208., 9289712., 9301216., 9312720., 9324224., 9335728.,
        9347232., 9358736., 9370240., 9381744., 9393248., 9404752., 9416256., 9427760., 9439264.,
        9450768., 9667328., 9679344., 9691360., 9703376., 9715392., 9727408., 9739424., 9751440.,
        9763456., 9775472., 9787488., 9799504., 9811520., 9823536., 9835552., 9847568.,
    ];

    assert_equals::<R>(results, expected, device);
}

/// Exported test
pub fn cmma_compute_loop_two_warps_same_tile_row_test<R: Runtime>(device: &R::Device) {
    if !cmma_available::<R>(device) {
        // We can't execute the test, skip.
        return;
    }

    let m = 16;
    let k = 32;
    let n = 64;

    let lhs = range_tensor_f16::<R>(m, k, device);
    let rhs = range_tensor_f16::<R>(k, n, device);
    let results = create_empty::<R>(m, n, device);
    let cube_dim = CubeDim::new(32, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = CmmaConfig {
        block_size_m: UInt::new(m as u32),
        block_size_k: UInt::new(k as u32),
        block_size_n: UInt::new(n as u32),
        tile_size: UInt::new(16),
        check_m_bounds: false,
        check_k_bounds: false,
        check_n_bounds: false,
        unroll: false,
    };

    compute_loop_test::launch::<F32, F16, R>(
        R::client(device),
        cube_count,
        cube_dim,
        TensorArg::new(&lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::new(&rhs.handle, &rhs.strides, &rhs.shape),
        ArrayArg::new(&results, m * n),
        UInt::new(m as u32),
        UInt::new(k as u32),
        UInt::new(n as u32),
        config,
    );

    let expected = &[
        1610496.0, 1614832.0, 1619168.0, 1623504.0, 1627840.0, 1632176.0, 1636512.0, 1640848.0,
        1645184.0, 1649520.0, 1653856.0, 1658192.0, 1662528.0, 1666864.0, 1671200.0, 1675536.0,
        1737472.0, 1742320.0, 1747168.0, 1752016.0, 1756864.0, 1761712.0, 1766560.0, 1771408.0,
        1776256.0, 1781104.0, 1785952.0, 1790800.0, 1795648.0, 1800496.0, 1805344.0, 1810192.0,
        1864448.0, 1869808.0, 1875168.0, 1880528.0, 1885888.0, 1891248.0, 1896608.0, 1901968.0,
        1907328.0, 1912688.0, 1918048.0, 1923408.0, 1928768.0, 1934128.0, 1939488.0, 1944848.0,
        1991424.0, 1997296.0, 2003168.0, 2009040.0, 2014912.0, 2020784.0, 2026656.0, 2032528.0,
        2038400.0, 2044272.0, 2050144.0, 2056016.0, 2061888.0, 2067760.0, 2073632.0, 2079504.0,
        2118400.0, 2124784.0, 2131168.0, 2137552.0, 2143936.0, 2150320.0, 2156704.0, 2163088.0,
        2169472.0, 2175856.0, 2182240.0, 2188624.0, 2195008.0, 2201392.0, 2207776.0, 2214160.0,
        2245376.0, 2252272.0, 2259168.0, 2266064.0, 2272960.0, 2279856.0, 2286752.0, 2293648.0,
        2300544.0, 2307440.0, 2314336.0, 2321232.0, 2328128.0, 2335024.0, 2341920.0, 2348816.0,
        2372352.0, 2379760.0, 2387168.0, 2394576.0, 2401984.0, 2409392.0, 2416800.0, 2424208.0,
        2431616.0, 2439024.0, 2446432.0, 2453840.0, 2461248.0, 2468656.0, 2476064.0, 2483472.0,
        2499328.0, 2507248.0, 2515168.0, 2523088.0, 2531008.0, 2538928.0, 2546848.0, 2554768.0,
        2562688.0, 2570608.0, 2578528.0, 2586448.0, 2594368.0, 2602288.0, 2610208.0, 2618128.0,
        2626304.0, 2634736.0, 2643168.0, 2651600.0, 2660032.0, 2668464.0, 2676896.0, 2685328.0,
        2693760.0, 2702192.0, 2710624.0, 2719056.0, 2727488.0, 2735920.0, 2744352.0, 2752784.0,
        2753280.0, 2762224.0, 2771168.0, 2780112.0, 2789056.0, 2798000.0, 2806944.0, 2815888.0,
        2824832.0, 2833776.0, 2842720.0, 2851664.0, 2860608.0, 2869552.0, 2878496.0, 2887440.0,
        2880256.0, 2889712.0, 2899168.0, 2908624.0, 2918080.0, 2927536.0, 2936992.0, 2946448.0,
        2955904.0, 2965360.0, 2974816.0, 2984272.0, 2993728.0, 3003184.0, 3012640.0, 3022096.0,
        3007232.0, 3017200.0, 3027168.0, 3037136.0, 3047104.0, 3057072.0, 3067040.0, 3077008.0,
        3086976.0, 3096944.0, 3106912.0, 3116880.0, 3126848.0, 3136816.0, 3146784.0, 3156752.0,
        3134208.0, 3144688.0, 3155168.0, 3165648.0, 3176128.0, 3186608.0, 3197088.0, 3207568.0,
        3218048.0, 3228528.0, 3239008.0, 3249488.0, 3259968.0, 3270448.0, 3280928.0, 3291408.0,
        3261184.0, 3272176.0, 3283168.0, 3294160.0, 3305152.0, 3316144.0, 3327136.0, 3338128.0,
        3349120.0, 3360112.0, 3371104.0, 3382096.0, 3393088.0, 3404080.0, 3415072.0, 3426064.0,
        3388160.0, 3399664.0, 3411168.0, 3422672.0, 3434176.0, 3445680.0, 3457184.0, 3468688.0,
        3480192.0, 3491696.0, 3503200.0, 3514704.0, 3526208.0, 3537712.0, 3549216.0, 3560720.0,
        3515136.0, 3527152.0, 3539168.0, 3551184.0, 3563200.0, 3575216.0, 3587232.0, 3599248.0,
        3611264.0, 3623280.0, 3635296.0, 3647312.0, 3659328.0, 3671344.0, 3683360.0, 3695376.0,
        3830528.0, 3834864.0, 3839200.0, 3843536.0, 3847872.0, 3852208.0, 3856544.0, 3860880.0,
        3865216.0, 3869552.0, 3873888.0, 3878224.0, 3882560.0, 3886896.0, 3891232.0, 3895568.0,
        4219648.0, 4224496.0, 4229344.0, 4234192.0, 4239040.0, 4243888.0, 4248736.0, 4253584.0,
        4258432.0, 4263280.0, 4268128.0, 4272976.0, 4277824.0, 4282672.0, 4287520.0, 4292368.0,
        4608768.0, 4614128.0, 4619488.0, 4624848.0, 4630208.0, 4635568.0, 4640928.0, 4646288.0,
        4651648.0, 4657008.0, 4662368.0, 4667728.0, 4673088.0, 4678448.0, 4683808.0, 4689168.0,
        4997888.0, 5003760.0, 5009632.0, 5015504.0, 5021376.0, 5027248.0, 5033120.0, 5038992.0,
        5044864.0, 5050736.0, 5056608.0, 5062480.0, 5068352.0, 5074224.0, 5080096.0, 5085968.0,
        5387008.0, 5393392.0, 5399776.0, 5406160.0, 5412544.0, 5418928.0, 5425312.0, 5431696.0,
        5438080.0, 5444464.0, 5450848.0, 5457232.0, 5463616.0, 5470000.0, 5476384.0, 5482768.0,
        5776128.0, 5783024.0, 5789920.0, 5796816.0, 5803712.0, 5810608.0, 5817504.0, 5824400.0,
        5831296.0, 5838192.0, 5845088.0, 5851984.0, 5858880.0, 5865776.0, 5872672.0, 5879568.0,
        6165248.0, 6172656.0, 6180064.0, 6187472.0, 6194880.0, 6202288.0, 6209696.0, 6217104.0,
        6224512.0, 6231920.0, 6239328.0, 6246736.0, 6254144.0, 6261552.0, 6268960.0, 6276368.0,
        6554368.0, 6562288.0, 6570208.0, 6578128.0, 6586048.0, 6593968.0, 6601888.0, 6609808.0,
        6617728.0, 6625648.0, 6633568.0, 6641488.0, 6649408.0, 6657328.0, 6665248.0, 6673168.0,
        6943488.0, 6951920.0, 6960352.0, 6968784.0, 6977216.0, 6985648.0, 6994080.0, 7002512.0,
        7010944.0, 7019376.0, 7027808.0, 7036240.0, 7044672.0, 7053104.0, 7061536.0, 7069968.0,
        7332608.0, 7341552.0, 7350496.0, 7359440.0, 7368384.0, 7377328.0, 7386272.0, 7395216.0,
        7404160.0, 7413104.0, 7422048.0, 7430992.0, 7439936.0, 7448880.0, 7457824.0, 7466768.0,
        7721728.0, 7731184.0, 7740640.0, 7750096.0, 7759552.0, 7769008.0, 7778464.0, 7787920.0,
        7797376.0, 7806832.0, 7816288.0, 7825744.0, 7835200.0, 7844656.0, 7854112.0, 7863568.0,
        8110848.0, 8120816.0, 8130784.0, 8140752.0, 8150720.0, 8160688.0, 8170656.0, 8180624.0,
        8190592.0, 8200560.0, 8210528.0, 8220496.0, 8230464.0, 8240432.0, 8250400.0, 8260368.0,
        8499968.0, 8510448.0, 8520928.0, 8531408.0, 8541888.0, 8552368.0, 8562848.0, 8573328.0,
        8583808.0, 8594288.0, 8604768.0, 8615248.0, 8625728.0, 8636208.0, 8646688.0, 8657168.0,
        8889088.0, 8900080.0, 8911072.0, 8922064.0, 8933056.0, 8944048.0, 8955040.0, 8966032.0,
        8977024.0, 8988016.0, 8999008.0, 9010000.0, 9020992.0, 9031984.0, 9042976.0, 9053968.0,
        9278208.0, 9289712.0, 9301216.0, 9312720.0, 9324224.0, 9335728.0, 9347232.0, 9358736.0,
        9370240.0, 9381744.0, 9393248.0, 9404752.0, 9416256.0, 9427760.0, 9439264.0, 9450768.0,
        9667328.0, 9679344.0, 9691360.0, 9703376.0, 9715392.0, 9727408.0, 9739424.0, 9751440.0,
        9763456.0, 9775472.0, 9787488.0, 9799504.0, 9811520.0, 9823536.0, 9835552.0, 9847568.0,
        6050560.0, 6054896.0, 6059232.0, 6063568.0, 6067904.0, 6072240.0, 6076576.0, 6080912.0,
        6085248.0, 6089584.0, 6093920.0, 6098256.0, 6102592.0, 6106928.0, 6111264.0, 6115600.0,
        6701824.0, 6706672.0, 6711520.0, 6716368.0, 6721216.0, 6726064.0, 6730912.0, 6735760.0,
        6740608.0, 6745456.0, 6750304.0, 6755152.0, 6760000.0, 6764848.0, 6769696.0, 6774544.0,
        7353088.0, 7358448.0, 7363808.0, 7369168.0, 7374528.0, 7379888.0, 7385248.0, 7390608.0,
        7395968.0, 7401328.0, 7406688.0, 7412048.0, 7417408.0, 7422768.0, 7428128.0, 7433488.0,
        8004352.0, 8010224.0, 8016096.0, 8021968.0, 8027840.0, 8033712.0, 8039584.0, 8045456.0,
        8051328.0, 8057200.0, 8063072.0, 8068944.0, 8074816.0, 8080688.0, 8086560.0, 8092432.0,
        8655616.0, 8662000.0, 8668384.0, 8674768.0, 8681152.0, 8687536.0, 8693920.0, 8700304.0,
        8706688.0, 8713072.0, 8719456.0, 8725840.0, 8732224.0, 8738608.0, 8744992.0, 8751376.0,
        9306880.0, 9313776.0, 9320672.0, 9327568.0, 9334464.0, 9341360.0, 9348256.0, 9355152.0,
        9362048.0, 9368944.0, 9375840.0, 9382736.0, 9389632.0, 9396528.0, 9403424.0, 9410320.0,
        9958144.0, 9965552.0, 9972960.0, 9980368.0, 9987776.0, 9995184.0, 10002592.0, 10010000.0,
        10017408.0, 10024816.0, 10032224.0, 10039632.0, 10047040.0, 10054448.0, 10061856.0,
        10069264.0, 10609408.0, 10617328.0, 10625248.0, 10633168.0, 10641088.0, 10649008.0,
        10656928.0, 10664848.0, 10672768.0, 10680688.0, 10688608.0, 10696528.0, 10704448.0,
        10712368.0, 10720288.0, 10728208.0, 11260672.0, 11269104.0, 11277536.0, 11285968.0,
        11294400.0, 11302832.0, 11311264.0, 11319696.0, 11328128.0, 11336560.0, 11344992.0,
        11353424.0, 11361856.0, 11370288.0, 11378720.0, 11387152.0, 11911936.0, 11920880.0,
        11929824.0, 11938768.0, 11947712.0, 11956656.0, 11965600.0, 11974544.0, 11983488.0,
        11992432.0, 12001376.0, 12010320.0, 12019264.0, 12028208.0, 12037152.0, 12046096.0,
        12563200.0, 12572656.0, 12582112.0, 12591568.0, 12601024.0, 12610480.0, 12619936.0,
        12629392.0, 12638848.0, 12648304.0, 12657760.0, 12667216.0, 12676672.0, 12686128.0,
        12695584.0, 12705040.0, 13214464.0, 13224432.0, 13234400.0, 13244368.0, 13254336.0,
        13264304.0, 13274272.0, 13284240.0, 13294208.0, 13304176.0, 13314144.0, 13324112.0,
        13334080.0, 13344048.0, 13354016.0, 13363984.0, 13865728.0, 13876208.0, 13886688.0,
        13897168.0, 13907648.0, 13918128.0, 13928608.0, 13939088.0, 13949568.0, 13960048.0,
        13970528.0, 13981008.0, 13991488.0, 14001968.0, 14012448.0, 14022928.0, 14516992.0,
        14527984.0, 14538976.0, 14549968.0, 14560960.0, 14571952.0, 14582944.0, 14593936.0,
        14604928.0, 14615920.0, 14626912.0, 14637904.0, 14648896.0, 14659888.0, 14670880.0,
        14681872.0, 15168256.0, 15179760.0, 15191264.0, 15202768.0, 15214272.0, 15225776.0,
        15237280.0, 15248784.0, 15260288.0, 15271792.0, 15283296.0, 15294800.0, 15306304.0,
        15317808.0, 15329312.0, 15340816.0, 15819520.0, 15831536.0, 15843552.0, 15855568.0,
        15867584.0, 15879600.0, 15891616.0, 15903632.0, 15915648.0, 15927664.0, 15939680.0,
        15951696.0, 15963712.0, 15975728.0, 15987744.0, 15999760.0, 8270592.0, 8274928.0,
        8279264.0, 8283600.0, 8287936.0, 8292272.0, 8296608.0, 8300944.0, 8305280.0, 8309616.0,
        8313952.0, 8318288.0, 8322624.0, 8326960.0, 8331296.0, 8335632.0, 9184000.0, 9188848.0,
        9193696.0, 9198544.0, 9203392.0, 9208240.0, 9213088.0, 9217936.0, 9222784.0, 9227632.0,
        9232480.0, 9237328.0, 9242176.0, 9247024.0, 9251872.0, 9256720.0, 10097408.0, 10102768.0,
        10108128.0, 10113488.0, 10118848.0, 10124208.0, 10129568.0, 10134928.0, 10140288.0,
        10145648.0, 10151008.0, 10156368.0, 10161728.0, 10167088.0, 10172448.0, 10177808.0,
        11010816.0, 11016688.0, 11022560.0, 11028432.0, 11034304.0, 11040176.0, 11046048.0,
        11051920.0, 11057792.0, 11063664.0, 11069536.0, 11075408.0, 11081280.0, 11087152.0,
        11093024.0, 11098896.0, 11924224.0, 11930608.0, 11936992.0, 11943376.0, 11949760.0,
        11956144.0, 11962528.0, 11968912.0, 11975296.0, 11981680.0, 11988064.0, 11994448.0,
        12000832.0, 12007216.0, 12013600.0, 12019984.0, 12837632.0, 12844528.0, 12851424.0,
        12858320.0, 12865216.0, 12872112.0, 12879008.0, 12885904.0, 12892800.0, 12899696.0,
        12906592.0, 12913488.0, 12920384.0, 12927280.0, 12934176.0, 12941072.0, 13751040.0,
        13758448.0, 13765856.0, 13773264.0, 13780672.0, 13788080.0, 13795488.0, 13802896.0,
        13810304.0, 13817712.0, 13825120.0, 13832528.0, 13839936.0, 13847344.0, 13854752.0,
        13862160.0, 14664448.0, 14672368.0, 14680288.0, 14688208.0, 14696128.0, 14704048.0,
        14711968.0, 14719888.0, 14727808.0, 14735728.0, 14743648.0, 14751568.0, 14759488.0,
        14767408.0, 14775328.0, 14783248.0, 15577856.0, 15586288.0, 15594720.0, 15603152.0,
        15611584.0, 15620016.0, 15628448.0, 15636880.0, 15645312.0, 15653744.0, 15662176.0,
        15670608.0, 15679040.0, 15687472.0, 15695904.0, 15704336.0, 16491264.0, 16500208.0,
        16509152.0, 16518096.0, 16527040.0, 16535984.0, 16544928.0, 16553872.0, 16562816.0,
        16571760.0, 16580704.0, 16589648.0, 16598592.0, 16607536.0, 16616480.0, 16625424.0,
        17404672.0, 17414128.0, 17423584.0, 17433040.0, 17442496.0, 17451952.0, 17461408.0,
        17470864.0, 17480320.0, 17489776.0, 17499232.0, 17508688.0, 17518144.0, 17527600.0,
        17537056.0, 17546512.0, 18318080.0, 18328048.0, 18338016.0, 18347984.0, 18357952.0,
        18367920.0, 18377888.0, 18387856.0, 18397824.0, 18407792.0, 18417760.0, 18427728.0,
        18437696.0, 18447664.0, 18457632.0, 18467600.0, 19231488.0, 19241968.0, 19252448.0,
        19262928.0, 19273408.0, 19283888.0, 19294368.0, 19304848.0, 19315328.0, 19325808.0,
        19336288.0, 19346768.0, 19357248.0, 19367728.0, 19378208.0, 19388688.0, 20144896.0,
        20155888.0, 20166880.0, 20177872.0, 20188864.0, 20199856.0, 20210848.0, 20221840.0,
        20232832.0, 20243824.0, 20254816.0, 20265808.0, 20276800.0, 20287792.0, 20298784.0,
        20309776.0, 21058304.0, 21069808.0, 21081312.0, 21092816.0, 21104320.0, 21115824.0,
        21127328.0, 21138832.0, 21150336.0, 21161840.0, 21173344.0, 21184848.0, 21196352.0,
        21207856.0, 21219360.0, 21230864.0, 21971712.0, 21983728.0, 21995744.0, 22007760.0,
        22019776.0, 22031792.0, 22043808.0, 22055824.0, 22067840.0, 22079856.0, 22091872.0,
        22103888.0, 22115904.0, 22127920.0, 22139936.0, 22151952.0,
    ];

    assert_equals::<R>(results, expected, device);
}
