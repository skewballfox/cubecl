/* automatically generated by rust-bindgen 0.71.0 */
/* modified to remove prefixes */

pub const Version: _bindgen_ty_1 = _bindgen_ty_1::Version;
pub const Version_BitWidthPadding: _bindgen_ty_1 = _bindgen_ty_1::Version_BitWidthPadding;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum _bindgen_ty_1 {
    Version = 100,
    Version_BitWidthPadding = 2147483647,
}
pub const Revision: _bindgen_ty_2 = _bindgen_ty_2::Revision;
pub const Revision_BitWidthPadding: _bindgen_ty_2 = _bindgen_ty_2::Revision_BitWidthPadding;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum _bindgen_ty_2 {
    Revision = 6,
    Revision_BitWidthPadding = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Instructions {
    DebugInfoNone = 0,
    DebugCompilationUnit = 1,
    DebugTypeBasic = 2,
    DebugTypePointer = 3,
    DebugTypeQualifier = 4,
    DebugTypeArray = 5,
    DebugTypeVector = 6,
    DebugTypedef = 7,
    DebugTypeFunction = 8,
    DebugTypeEnum = 9,
    DebugTypeComposite = 10,
    DebugTypeMember = 11,
    DebugTypeInheritance = 12,
    DebugTypePtrToMember = 13,
    DebugTypeTemplate = 14,
    DebugTypeTemplateParameter = 15,
    DebugTypeTemplateTemplateParameter = 16,
    DebugTypeTemplateParameterPack = 17,
    DebugGlobalVariable = 18,
    DebugFunctionDeclaration = 19,
    DebugFunction = 20,
    DebugLexicalBlock = 21,
    DebugLexicalBlockDiscriminator = 22,
    DebugScope = 23,
    DebugNoScope = 24,
    DebugInlinedAt = 25,
    DebugLocalVariable = 26,
    DebugInlinedVariable = 27,
    DebugDeclare = 28,
    DebugValue = 29,
    DebugOperation = 30,
    DebugExpression = 31,
    DebugMacroDef = 32,
    DebugMacroUndef = 33,
    DebugImportedEntity = 34,
    DebugSource = 35,
    DebugFunctionDefinition = 101,
    DebugSourceContinued = 102,
    DebugLine = 103,
    DebugNoLine = 104,
    DebugBuildIdentifier = 105,
    DebugStoragePath = 106,
    DebugEntryPoint = 107,
    DebugTypeMatrix = 108,
    InstructionsMax = 2147483647,
}
impl DebugInfoFlags {
    pub const None: DebugInfoFlags = DebugInfoFlags(0);
}
impl DebugInfoFlags {
    pub const FlagIsProtected: DebugInfoFlags = DebugInfoFlags(1);
}
impl DebugInfoFlags {
    pub const FlagIsPrivate: DebugInfoFlags = DebugInfoFlags(2);
}
impl DebugInfoFlags {
    pub const FlagIsPublic: DebugInfoFlags = DebugInfoFlags(3);
}
impl DebugInfoFlags {
    pub const FlagIsLocal: DebugInfoFlags = DebugInfoFlags(4);
}
impl DebugInfoFlags {
    pub const FlagIsDefinition: DebugInfoFlags = DebugInfoFlags(8);
}
impl DebugInfoFlags {
    pub const FlagFwdDecl: DebugInfoFlags = DebugInfoFlags(16);
}
impl DebugInfoFlags {
    pub const FlagArtificial: DebugInfoFlags = DebugInfoFlags(32);
}
impl DebugInfoFlags {
    pub const FlagExplicit: DebugInfoFlags = DebugInfoFlags(64);
}
impl DebugInfoFlags {
    pub const FlagPrototyped: DebugInfoFlags = DebugInfoFlags(128);
}
impl DebugInfoFlags {
    pub const FlagObjectPointer: DebugInfoFlags = DebugInfoFlags(256);
}
impl DebugInfoFlags {
    pub const FlagStaticMember: DebugInfoFlags = DebugInfoFlags(512);
}
impl DebugInfoFlags {
    pub const FlagIndirectVariable: DebugInfoFlags = DebugInfoFlags(1024);
}
impl DebugInfoFlags {
    pub const FlagLValueReference: DebugInfoFlags = DebugInfoFlags(2048);
}
impl DebugInfoFlags {
    pub const FlagRValueReference: DebugInfoFlags = DebugInfoFlags(4096);
}
impl DebugInfoFlags {
    pub const FlagIsOptimized: DebugInfoFlags = DebugInfoFlags(8192);
}
impl DebugInfoFlags {
    pub const FlagIsEnumClass: DebugInfoFlags = DebugInfoFlags(16384);
}
impl DebugInfoFlags {
    pub const FlagTypePassByValue: DebugInfoFlags = DebugInfoFlags(32768);
}
impl DebugInfoFlags {
    pub const FlagTypePassByReference: DebugInfoFlags = DebugInfoFlags(65536);
}
impl DebugInfoFlags {
    pub const FlagUnknownPhysicalLayout: DebugInfoFlags = DebugInfoFlags(131072);
}
impl DebugInfoFlags {
    pub const DebugInfoFlagsMax: DebugInfoFlags = DebugInfoFlags(2147483647);
}
impl ::std::ops::BitOr<DebugInfoFlags> for DebugInfoFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        DebugInfoFlags(self.0 | other.0)
    }
}
impl ::std::ops::BitOrAssign for DebugInfoFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: DebugInfoFlags) {
        self.0 |= rhs.0;
    }
}
impl ::std::ops::BitAnd<DebugInfoFlags> for DebugInfoFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        DebugInfoFlags(self.0 & other.0)
    }
}
impl ::std::ops::BitAndAssign for DebugInfoFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: DebugInfoFlags) {
        self.0 &= rhs.0;
    }
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct DebugInfoFlags(pub u32);
impl BuildIdentifierFlags {
    pub const IdentifierPossibleDuplicates: BuildIdentifierFlags = BuildIdentifierFlags(1);
}
impl BuildIdentifierFlags {
    pub const BuildIdentifierFlagsMax: BuildIdentifierFlags = BuildIdentifierFlags(2147483647);
}
impl ::std::ops::BitOr<BuildIdentifierFlags> for BuildIdentifierFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        BuildIdentifierFlags(self.0 | other.0)
    }
}
impl ::std::ops::BitOrAssign for BuildIdentifierFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: BuildIdentifierFlags) {
        self.0 |= rhs.0;
    }
}
impl ::std::ops::BitAnd<BuildIdentifierFlags> for BuildIdentifierFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        BuildIdentifierFlags(self.0 & other.0)
    }
}
impl ::std::ops::BitAndAssign for BuildIdentifierFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: BuildIdentifierFlags) {
        self.0 &= rhs.0;
    }
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct BuildIdentifierFlags(pub u32);
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DebugBaseTypeAttributeEncoding {
    Unspecified = 0,
    Address = 1,
    Boolean = 2,
    Float = 3,
    Signed = 4,
    SignedChar = 5,
    Unsigned = 6,
    UnsignedChar = 7,
    DebugBaseTypeAttributeEncodingMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DebugCompositeType {
    Class = 0,
    Structure = 1,
    Union = 2,
    DebugCompositeTypeMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DebugTypeQualifier {
    ConstType = 0,
    VolatileType = 1,
    RestrictType = 2,
    AtomicType = 3,
    DebugTypeQualifierMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DebugOperation {
    Deref = 0,
    Plus = 1,
    Minus = 2,
    PlusUconst = 3,
    BitPiece = 4,
    Swap = 5,
    Xderef = 6,
    StackValue = 7,
    Constu = 8,
    Fragment = 9,
    DebugOperationMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DebugImportedEntity {
    ImportedModule = 0,
    ImportedDeclaration = 1,
    DebugImportedEntityMax = 2147483647,
}
