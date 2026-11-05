/// Configuration for persistent kernels in `CubeCL`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PersistentConfig {
    /// The part of the device capacity that [`PersistentCount::Fill`] uses, in `(0, 1]`.
    ///
    /// Lower it to leave room for kernels on other streams. A value outside the range gives `1.0`.
    ///
    /// [`PersistentCount::Fill`]: crate::persistent::PersistentCount::Fill
    #[serde(default = "default_fill_fraction")]
    pub fill_fraction: f32,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            fill_fraction: default_fill_fraction(),
        }
    }
}

fn default_fill_fraction() -> f32 {
    1.0
}
