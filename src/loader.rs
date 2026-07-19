//! Asset loader for `.inx`/`.inp` (feature `inx`) - the authoring IR format.
//!
//! Parses the authoring JSON (`inochi2d_parser::owned::Puppet`), converts it in
//! memory into the same typed INR document + binary blob the exporter produces
//! (`inochi2d_parser::inr::convert_puppet` - no JSON round-trip, no file written),
//! and hands that to `inr_loader::convert`, the exact conversion `.inr` files go
//! through. Sharing that path means the two loaders produce identical results
//! (including baked mask contours, which only `convert_puppet` computes).
//!
//! Texture decoding (PNG/TGA to raw RGBA) happens here at load time, the same cost
//! as pre-exporting to `.inr` - a transient RAM peak during the load, not retained
//! afterward.

use bevy::asset::{AssetLoader, LoadContext};
use bevy::reflect::TypePath;
use inochi2d_parser::inr::{self, InrModel};
use inochi2d_parser::owned::Puppet as RawPuppet;

use crate::InxPuppet;

/// `AssetLoader` for `.inx`/`.inp`, registered only with feature `inx`.
#[derive(TypePath)]
pub struct InxLoader;

impl AssetLoader for InxLoader {
    type Asset = InxPuppet;

    type Settings = ();

    type Error = inr::InrError;

    async fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| inr::InrError::Truncated)?;

        let raw = RawPuppet::from_bytes(&bytes)?;
        let (doc, bin) = inr::convert_puppet(&raw)?;
        let model = InrModel { doc, bin };
        crate::inr_loader::convert(&model, load_context)
    }

    fn extensions(&self) -> &[&str] {
        &["inx", "inp"]
    }
}
