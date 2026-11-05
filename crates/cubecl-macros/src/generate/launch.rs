use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{GenericParam, Generics, Ident, parse_quote};

use crate::{
    parse::{
        kernel::{
            AddressType, ExecutionMode, GenericArg, Launch, anon_lifetime_to_static, strip_ref,
        },
        signature::KernelParam,
    },
    paths::{core_type, prelude_type},
};

impl ToTokens for Launch {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.vis;

        let name = &self.func.sig.name;
        let launch = self.launch();
        let aliases = self.create_type_alias();
        let dummy = self.create_dummy_kernel();
        let kernel = self.kernel_definition();
        let mut func = self.func.clone();
        func.sig.name = format_ident!("expand");
        let func = func.to_tokens_mut();

        let out = quote! {
            #vis mod #name {
                use super::*;

                #aliases

                #[allow(unused, clippy::all)]
                #func

                #kernel
                #launch
                #dummy
            }
        };

        if self.args.debug.is_present() {
            let file = syn::parse_file(&out.to_string()).unwrap();
            let tokens = prettyplease::unparse(&file);
            panic!("{tokens}");
        }
        tokens.extend(out);
    }
}

/// The cube count a generated launch function takes, and how it launches the kernel.
enum LaunchCount {
    /// A `CubeCount` from the caller.
    Grid,
    /// A `PersistentCount`, with the capacity from the default hint.
    Persistent,
    /// A `PersistentCount`, with the capacity from a hint generic.
    PersistentWith,
}

impl Launch {
    fn launch(&self) -> TokenStream {
        let mut out = TokenStream::new();
        // A persistent kernel launches only through the runtime, which sets the count and binds
        // the launch workspace.
        let persistent = self.args.is_persistent();
        for (flag, mode, suffix) in [
            (&self.args.launch, ExecutionMode::Checked, ""),
            (
                &self.args.launch_unchecked,
                ExecutionMode::Unchecked,
                "_unchecked",
            ),
        ] {
            if !flag.is_present() {
                continue;
            }
            if !persistent {
                out.extend(self.launch_fn(
                    format_ident!("launch{suffix}"),
                    mode,
                    LaunchCount::Grid,
                    false,
                ));
            }
            if persistent {
                for (name, count) in [
                    ("launch_persistent", LaunchCount::Persistent),
                    ("launch_persistent_with", LaunchCount::PersistentWith),
                ] {
                    out.extend(self.launch_fn(format_ident!("{name}{suffix}"), mode, count, false));
                }
                out.extend(self.capacity_fn(format_ident!("capacity{suffix}"), mode));
            }
            // Only a grid sync needs the device to itself (D9).
            if self.args.cooperative.is_present() {
                for (name, count) in [
                    ("launch_persistent_exclusive", LaunchCount::Persistent),
                    (
                        "launch_persistent_exclusive_with",
                        LaunchCount::PersistentWith,
                    ),
                ] {
                    out.extend(self.launch_fn(format_ident!("{name}{suffix}"), mode, count, true));
                }
            }
        }
        out
    }

    fn launch_fn(
        &self,
        name: Ident,
        mode: ExecutionMode,
        count: LaunchCount,
        exclusive: bool,
    ) -> TokenStream {
        let compute_client = prelude_type("Client");
        let cube_dim = prelude_type("CubeDim");
        let kernel_name = &self.func.sig.name;
        let mut doc = format!("Launch the kernel [{kernel_name}()] on the given runtime");
        if !matches!(count, LaunchCount::Grid) {
            doc.push_str(" as a persistent kernel");
        }
        if exclusive {
            doc.push_str(", with native grid sync on a device that other work shares");
        }
        if matches!(mode, ExecutionMode::Unchecked) {
            doc.push_str(" without bound checks");
        }
        let mut generics = self.launch_fn_generics(mode);

        let (count_param, launch) = match count {
            LaunchCount::Grid => {
                let cube_count = prelude_type("CubeCount");
                (
                    quote![__cube_count: #cube_count],
                    quote![launcher.launch(__cube_count, __kernel, __client)],
                )
            }
            LaunchCount::Persistent | LaunchCount::PersistentWith => {
                let persistent_count = prelude_type("PersistentCount");
                let hint = match count {
                    LaunchCount::PersistentWith => {
                        let capacity_hint = prelude_type("CapacityHint");
                        generics.params.push(parse_quote![__H: #capacity_hint]);
                        doc.push_str(
                            ".\n\n`__H` estimates the capacity on a runtime that cannot query it.",
                        );
                        quote![__H]
                    }
                    _ => prelude_type("DefaultCapacity").into_token_stream(),
                };
                let launch = if exclusive {
                    quote![unsafe {
                        launcher.launch_persistent_exclusive::<#hint, _>(__count, __kernel, __client)
                    }]
                } else {
                    quote![launcher.launch_persistent::<#hint, _>(__count, __kernel, __client)]
                };
                (quote![__count: #persistent_count], launch)
            }
        };
        let unsafety = self.safety_doc(mode, exclusive, &mut doc);
        let args = self.launch_args();
        let address_type = self.address_type_param();
        let body = self.launch_body(mode);

        quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub #unsafety fn #name #generics(
                __client: &#compute_client,
                #count_param,
                __cube_dim: #cube_dim,
                #address_type
                #(#args),*
            ) {
                #body
                #launch
            }
        }
    }

    /// How many cubes of the kernel the device runs at the same time.
    fn capacity_fn(&self, name: Ident, mode: ExecutionMode) -> TokenStream {
        let compute_client = prelude_type("Client");
        let cube_dim = prelude_type("CubeDim");
        let server_error = prelude_type("ServerError");
        let doc = format!(
            "How many cubes of the kernel [{}()] the device runs at the same time, or `None` if \
             the runtime cannot query it.",
            self.func.sig.name
        );
        let generics = self.launch_fn_generics(mode);
        let args = self.launch_args();
        let address_type = self.address_type_param();
        let body = self.launch_body(mode);

        quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn #name #generics(
                __client: &#compute_client,
                __cube_dim: #cube_dim,
                #address_type
                #(#args),*
            ) -> Result<Option<u32>, #server_error> {
                #body
                launcher.capacity(__kernel, __client)
            }
        }
    }

    fn launch_fn_generics(&self, mode: ExecutionMode) -> Generics {
        match mode {
            ExecutionMode::Checked => self.launch_generics.clone(),
            ExecutionMode::Unchecked => self.kernel_generics.clone(),
        }
    }

    fn address_type_param(&self) -> TokenStream {
        let address_type = prelude_type("AddressType");
        match self.args.address_type {
            AddressType::Dynamic => quote![__address_type: #address_type,],
            _ => quote![],
        }
    }

    /// Appends the `# Safety` section to `doc`, and returns `unsafe` if the function needs it.
    fn safety_doc(&self, mode: ExecutionMode, exclusive: bool, doc: &mut String) -> TokenStream {
        let mut rules = Vec::new();
        if matches!(mode, ExecutionMode::Unchecked) {
            rules.push("Contain any out of bounds reads or writes. Doing so is immediate UB.");
            rules.push(
                "Contain any loops that never terminate. These may be optimized away entirely or \
                 cause\n  other unpredictable behaviour.",
            );
        }
        if self.args.may_hang() {
            rules.push(
                "Run on a device that does not run all cubes of the launch at the same time. \
                 The `spin` grid sync then hangs.",
            );
        }
        if rules.is_empty() && !exclusive {
            return TokenStream::new();
        }
        if !doc.ends_with('.') {
            doc.push('.');
        }
        doc.push_str("\n\n# Safety\n");
        if !rules.is_empty() {
            doc.push_str("\nThe kernel must not:\n");
        }
        for rule in rules {
            doc.push_str(&format!("- {rule}\n"));
        }
        if exclusive {
            doc.push_str(
                "\nNo other work may use the compute units of the device during the launch. \
                 This includes a display and other processes. Else some cubes cannot start, the \
                 grid sync deadlocks, and the driver resets the device. A reset destroys every \
                 context on the device.\n",
            );
        }
        quote![unsafe]
    }

    fn launch_body(&self, execution_mode: ExecutionMode) -> TokenStream {
        let kernel_name = self.kernel_name();
        let kernel_generics = self.kernel_call_generics();
        let kernel_generics = kernel_generics.split_for_impl();
        let kernel_generics = kernel_generics.1.as_turbofish();
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let (setup, args) = self.launcher_setup(execution_mode, quote![__client.properties()]);

        quote! {
            #setup
            let __kernel = #kernel_name #kernel_generics::new(
                __settings,
                __client.properties_shared(),
                __client.target_properties_shared(),
                #args #(#comptime_args),*
            );
        }
    }

    /// The statements every launch-shaped function opens with: `__settings`,
    /// a `launcher` whose scope has the generic types registered, and one
    /// `comp_arg_*` local per runtime argument. `device_properties` is the
    /// `&DeviceProperties` expression the scope reads, since the real launch
    /// has a client and the dummy kernel has the properties in hand.
    ///
    /// Returned with the argument list the kernel's constructor takes, so the
    /// registration that produces a kernel's compilation arguments, and with
    /// them its `KernelId`, is written once.
    fn launcher_setup(
        &self,
        execution_mode: ExecutionMode,
        device_properties: TokenStream,
    ) -> (TokenStream, TokenStream) {
        let kernel_launcher = prelude_type("KernelLauncher");

        let mappings = self.func.sig.define_mappings();
        let generic_registers =
            self.func
                .analysis
                .register_types(mappings, quote![scope], false, true);
        let settings = self.configure_settings(execution_mode);
        let (registers, args) = self.arg_registers();

        let setup = quote! {
            #settings

            let mut launcher = #kernel_launcher::new(__settings.clone());
            launcher.with_scope(|scope| {
                scope.device_properties(#device_properties);
                #generic_registers
            });

            #registers
        };

        (setup, args)
    }

    fn configure_settings(&self, mode: ExecutionMode) -> TokenStream {
        let kernel_settings = prelude_type("KernelSettings");
        let addr_ty = prelude_type("AddressType");
        let address_type = match self.args.address_type {
            AddressType::U32 => quote![#addr_ty::U32],
            AddressType::U64 => quote![#addr_ty::U64],
            AddressType::Dynamic => quote![__address_type],
        };

        quote! {
            let mut __settings = #kernel_settings::new(__cube_dim.into(), #mode, #address_type);
        }
    }

    fn create_type_alias(&self) -> TokenStream {
        let mut aliases = quote! {};
        if !self.func.args.explicit_define.is_present() {
            for (name, GenericArg { expand_ty, .. }) in self.func.analysis.map.iter() {
                aliases.extend(quote! {
                    /// Type to be used as a generic for launch kernel argument.
                    pub type #name = #expand_ty;
                });
            }
        }

        aliases
    }
    fn create_dummy_kernel(&self) -> TokenStream {
        if self.args.create_dummy_kernel.is_present() {
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let device_properties = prelude_type("DeviceProperties");
            let target_properties = prelude_type("TargetProperties");
            let private = core_type("__private");
            let generics = &self.kernel_generics;
            let (_, generic_names, _) = self.kernel_generics.split_for_impl();

            let kernel_name = self.kernel_name();
            let comptime_args = self.launch_args();
            let comptime_names = self.comptime_params().map(|it| &it.name);
            let (setup, args) =
                self.launcher_setup(ExecutionMode::Checked, quote![&__device_properties]);

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn create_dummy_kernel #generics(
                    __device_properties: #private::Arc<#device_properties>,
                    __target_properties: #private::Arc<#target_properties>,
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #address_type
                    #(#comptime_args),*
                ) -> #kernel_name #generic_names {
                    // The same registration a launch does, so the kernel's
                    // compilation arguments, and with them its id, come out
                    // identical. The launcher has nothing to launch here, so
                    // it is discarded rather than dropped: see
                    // `KernelLauncher::discard`.
                    #setup
                    launcher.discard();

                    #kernel_name::new(__settings, __device_properties, __target_properties, #args #(#comptime_names),*)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func.sig.runtime_params()
    }

    fn launch_args(&self) -> Vec<KernelParam> {
        let mut args = self.func.sig.parameters.clone();
        let runtime_arg = core_type("RuntimeArg");
        for arg in args.iter_mut().filter(|it| !it.is_const) {
            let ty = strip_ref(arg.ty.clone());
            let ty = anon_lifetime_to_static(ty);
            arg.normalized_ty = parse_quote![#runtime_arg<#ty>];
            arg.mutability = None;
        }
        args
    }

    pub fn kernel_name(&self) -> Ident {
        let kernel_name = RenameRule::PascalCase.apply_to_field(self.func.sig.name.to_string());
        format_ident!("{kernel_name}")
    }

    pub fn kernel_call_generics(&self) -> Generics {
        let mut generics = self.kernel_generics.clone();
        generics.params = generics
            .params
            .into_iter()
            .filter(|it| !matches!(it, GenericParam::Lifetime(..)))
            .collect();
        generics
    }

    pub fn comptime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func
            .sig
            .parameters
            .iter()
            .filter(|param| param.is_const)
    }
}
