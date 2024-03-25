use ff::PrimeField;
use group::Group;

/// Wrapper struct around a field element that implements additional traits
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FWrap<F>(pub F);

impl<F: PrimeField> Copy for FWrap<F> {}

/// Wrapper struct around a Group element that implements additional traits
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GWrap<G>(pub G);

impl<G: Group> Copy for GWrap<G> {}
